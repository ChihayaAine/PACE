# PACE: Predictive Adaptive Context Extraction for Long-Horizon LLM Agents

Large Language Model (LLM) agents struggle with ultra-long-horizon tasks requiring hundreds or thousands of interaction steps. Traditional context management approaches face a fundamental dilemma: preserving complete histories rapidly exhausts context windows and forces crude truncation, while aggressive summarization discards critical information prematurely. We propose Predictive Adaptive Context Extraction (PACE), a novel framework that reconceptualizes context management as a Next Step Prediction problem. Inspired by neural attention, PACE dynamically constructs context by adjusting historical memory granularity based on its predicted relevance for the next action. Comprehensive evaluation across diverse benchmarks and models demonstrates that PACE consistently improves task success rates, with larger gains on complex tasks and robust cross-lingual performance. Crucially, PACE enables agents to sustain effective reasoning for 4,897 interaction steps in ultra-long-horizon scenarios, achieving a 66.2× improvement over the full-context ReAct baseline and 5.1× over advanced folding baselines. This fundamentally advances the capability of LLM-based agents in previously intractable long-horizon scenarios.

## Contributions

- We introduce PACE, a novel framework that reframes context management as a Next Step Prediction problem, dynamically constructing context through vectorized attention and pressure-adaptive compression based on predicted relevance for the immediate next action.

- We demonstrate comprehensive improvements across six diverse benchmarks and four models of varying scales, including state-of-the-art web agents. PACE achieves larger gains on complex tasks compared to simpler ones, while maintaining robust performance across languages and domains.

- We significantly extend agent operational longevity, achieving 4,897 interaction steps in ultra-long-horizon scenarios, representing a 66.2× improvement over the full-context ReAct baseline and 5.1× over advanced folding baselines.

## PACE 

PACE's architecture comprises five coordinated components. The **Main Agent** is the core decision-maker, receiving the dynamically constructed context $\mathcal{C}_t$ to generate actions using a state-of-the-art LLM. The **External Memory Store** ($\mathcal{M}$) maintains the complete interaction history: $\mathcal{M}_t = \{\text{Chunk}_1, \ldots, \text{Chunk}_{t-1}\}$. The **Representation Generator** utilizes a LLM (e.g., Gemini 2.5 Flash-Lite) to asynchronously produce multi-level summaries for each chunk, avoiding blockage of the main agent loop. The **Attention Scorer** employs a locally-hosted dense retrieval model (e.g., BGE-M3) to predict chunk relevance via semantic similarity. Finally, the **Context Builder** synthesizes relevance scores with adaptive compression policies to assemble $\mathcal{C}_t$ within token budget constraints.

Each interaction chunk maintains four representations at different granularity levels:

$$
\text{Chunk}_i = \{ id_i, \, type_i, \, t_i, \, R_{\text{full}}^{(i)}, \, R_{\text{detailed}}^{(i)}, R_{\text{brief}}^{(i)}, \, R_{\text{ph}}^{(i)}, \, \mathbf{k}_i \}
$$

where $\mathbf{k}_i$ is the pre-computed key embedding. The representations range from complete content ($R_{\text{full}}$), a comprehensive summary ($R_{\text{detailed}}$), a concise 1-2 sentence summary ($R_{\text{brief}}$), to a minimal placeholder ($R_{\text{ph}}$). These multiple granularity levels enable flexible compression adapting to varying relevance scores.

![PACE FrameWork](resource/fig1.png)

## Predictive Attention with Adaptive Compression

PACE's core innovation treats historical context selection as a next-step prediction problem. Let $N$ denote the number of recent chunks preserved in full detail (we set $N=2$). Given the user's query $Q$ and the most recent $N$ chunks, we construct their concatenation $\mathcal{R}_t = Q \oplus \bigoplus_{j=t-N}^{t-1} R_{\text{full}}^{(j)}$. We encode both the query and historical chunks using an embedding model, which produces dense vectors:

$$
\begin{aligned}
\mathbf{q}_t &= \text{Enc}(\text{Truncate}(\mathcal{R}_t, L_{\max})) \\
\mathbf{k}_i &= \text{Enc}(\text{Truncate}(R_{\text{full}}^{(i)}, L_{\max}))
\end{aligned}
$$

where $L_{\max}$ is the encoder's maximum input length, and $\mathbf{k}_i$ is the key embedding for each historical chunk $\text{Chunk}_i$ where $i \leq t-N-1$. This symmetric truncation strategy ensures both queries and keys fully utilize the encoder's capacity while maintaining balanced semantic representation. Key embeddings are computed once at chunk creation time and cached, enabling efficient retrieval even before detailed summaries are generated asynchronously. The summaries ($R_{\text{detailed}}$, $R_{\text{brief}}$) are used solely for multi-granularity presentation in the final context.

Let $M = t - N - 1$ denote the number of chunks requiring scoring. Raw similarity scores are computed via cosine similarity: $s_i = \cos(\mathbf{q}_t, \mathbf{k}_i)$. We then apply softmax with low temperature $\tau = 0.3$ to sharpen the distribution, accentuating relevant chunks while suppressing irrelevant ones:

$$
w_i = \frac{\exp(s_i / \tau)}{\sum_{j=1}^{M} \exp(s_j / \tau)}
$$

To enable adaptive thresholding, we compute relative weights: $\tilde{w}_i = M \cdot w_i$. A relative weight $\tilde{w}_i > 1$ indicates above-average relevance. With low temperature $\tau$, the softmax becomes highly peaked, concentrating weight on top-ranked chunks while driving most $\tilde{w}_i$ values well below 1, naturally facilitating intensified compression.

**Pressure-Adaptive Thresholds.** A key PACE feature is adapting compression intensity based on interaction state. We define "compression pressure" $P_t \in [0, 1]$ reflecting both task progression and context budget utilization:

$$
P_t = \max\left(\frac{t}{T_{\max}}, \frac{|\mathcal{C}_{t-1}|}{B_{\max}}\right)
$$

where $T_{\max}$ is the expected maximum task length, $B_{\max}$ is the token budget (set to 128K), and $|\mathcal{C}_{t-1}|$ denotes the previous context's token count. In practice, $T_{\max}$ is set to the 95th percentile of observed task lengths in each benchmark's validation set, providing a robust estimate without requiring precise horizon knowledge. We initialize $|\mathcal{C}_0|$ as the system prompt plus query length, ensuring well-defined initial conditions and avoiding circular dependency.

Base thresholds $(\alpha_0, \beta_0, \gamma_0) = (0.4, 0.8, 1.5)$ are adjusted dynamically: $\alpha_t = \alpha_0 \cdot (1 + \lambda P_t)$, and similarly for $\beta_t$ and $\gamma_t$, where $\lambda = 0.5$ controls adaptation rate. As pressure $P_t$ increases, thresholds rise proportionally, making it progressively harder for chunks to qualify for detailed representations. This pushes more chunks toward brief summaries or placeholders, intensifying compression automatically as context accumulates. The adjusted relative weight $\tilde{w}_i$ determines the representation level:

$$
\text{Select}(\text{Chunk}_i) = \begin{cases}
R_{\text{full}}^{(i)} & \text{if } \tilde{w}_i > \gamma_t \\
R_{\text{detailed}}^{(i)} & \text{if } \beta_t < \tilde{w}_i \leq \gamma_t \\
R_{\text{brief}}^{(i)} & \text{if } \alpha_t < \tilde{w}_i \leq \beta_t \\
R_{\text{ph}}^{(i)} & \text{if } \tilde{w}_i \leq \alpha_t
\end{cases}
$$

The final context is assembled as:

$$
\mathcal{C}_t = \text{Sys} \oplus Q \oplus \left(\bigoplus_{i=1}^{M} \text{Sel}_i\right) \oplus \left(\bigoplus_{j=t-N}^{t-1} R_{\text{full}}^{(j)}\right)
$$

where $\text{Sys}$ denotes the system prompt and $\text{Sel}_i = \text{Select}(\text{Chunk}_i)$.

## Ultra-Long Horizon Stress Test

PACE with its standard adaptive pressure ($\lambda=0.5$) sustained operation for 2,776 steps---a 2.9× improvement in operational longevity over the Folding Agent and a remarkable 37.5× improvement over the full-context ReAct baseline. Even more remarkably, by increasing the adaptation rate for aggressive compression ($\lambda=1.0$), PACE extended its operational horizon to 4,897 steps, achieving a 5.1× improvement over Folding Agent and an exceptional 66.2× improvement over ReAct. This result powerfully demonstrates that PACE's predictive, pressure-adaptive mechanism is critical for navigating ultra-long-horizon tasks, enabling agents to maintain coherent reasoning over thousands of interaction steps within a finite, albeit large, context window.

![Stress Test](resource/fig2.png)
