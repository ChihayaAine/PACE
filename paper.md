\title{PACE: Predictive Adaptive Context Extraction for Long-Horizon LLM Agents}

\author{First Author \\
  Affiliation / Address line 1 \\
  Affiliation / Address line 2 \\
  Affiliation / Address line 3 \\
  \texttt{email@domain} \\\And
  Second Author \\
  Affiliation / Address line 1 \\
  Affiliation / Address line 2 \\
  Affiliation / Address line 3 \\
  \texttt{email@domain} \\}

\begin{document}
\maketitle
\begin{abstract} 
Large Language Model (LLM) agents struggle with ultra-long-horizon tasks requiring hundreds or thousands of interaction steps. Traditional context management approaches face a fundamental dilemma: preserving complete histories rapidly exhausts context windows and forces crude truncation, while aggressive summarization discards critical information prematurely. We propose Predictive Adaptive Context Extraction (PACE), a novel framework that reconceptualizes context management as a Next Step Prediction problem. Inspired by neural attention, PACE dynamically constructs context by adjusting historical memory granularity based on its predicted relevance for the next action. Comprehensive evaluation across diverse benchmarks and models demonstrates that PACE consistently improves task success rates, with larger gains on complex tasks and robust cross-lingual performance. Crucially, PACE enables agents to sustain effective reasoning for 4,897 interaction steps in ultra-long-horizon scenarios, achieving a 66.2$\times$ improvement over the full-context ReAct baseline and 5.1$\times$ over advanced folding baselines. This fundamentally advances the capability of LLM-based agents in previously intractable long-horizon scenarios. Our code and data are available at \url{https://anonymous.4open.science/r/PACE-B000/}. 
\end{abstract}

\section{Introduction}
The deployment of Large Language Models (LLMs) as autonomous agents has revolutionized how AI systems tackle complex, multi-step tasks across diverse domains \citep{wei2022chain,yao2022react,schick2023toolformer,chowdhery2023palm,wu2023autogenenablingnextgenllm,xi2023risepotentiallargelanguage}.
These agent systems demonstrate remarkable capabilities in reasoning, tool use, and decision-making by maintaining context across extended interactions \citep{shinn2023reflexion,sun2025scaling,ye2025agentfoldlonghorizonwebagents,li2023camel,xie2023openagentsopenplatformlanguage,yang2025contextagentcontextawareproactivellm}.
\begin{figure*}[t]
  \includegraphics[width=\textwidth]{latex/PACE.pdf}
  \caption{An overview of our PACE framework.}
  \label{fig:experiments}
\end{figure*}

However, as task complexity and interaction length increase, a fundamental challenge emerges: the effective management of growing conversational context within the constraints of finite context windows. This limitation becomes increasingly severe in long-horizon tasks that may span hundreds or even thousands of interaction steps, where even models with extended context windows struggle to maintain decision quality as the history accumulates \citep{gajjar2025telellmhubbuildingcontextawaremultiagent,zhang2024surveymemorymechanismlarge}.
Traditional approaches to context management in agent systems face significant limitations. ReAct-style methods \citep{yao2022react,belcak2025smalllanguagemodelsfuture}, by preserving complete interaction histories, introduce high levels of redundancy that quickly exhaust the context window, forcing a crude truncation of older, potentially vital information. Step-wise summarization approaches \citep{jiang2023llmlinguacompressingpromptsaccelerated,xiao2023efficient} compress history at each step but often discard relevant information prematurely \citep{chhikara2025mem0buildingproductionreadyai}. While recent methods such as AgentFold have introduced specialized folding mechanisms, their utility is often tied to fine-tuning the agent to generate explicit `fold` instructions. This approach intertwines the primary task-solving process with a secondary, learned meta-cognitive skill of memory management, which can limit the model's applicability and general-purpose reasoning capabilities \citep{li2025sculptorempoweringllmscognitive}. This dependency, alongside existing context management issues, becomes particularly problematic in ultra-long-horizon scenarios \citep{mei2025surveycontextengineeringlarge}. In such cases, agents may lose track of critical information from earlier steps, fail to synthesize insights across distant interactions, or become overwhelmed by irrelevant details that dilute decision-making quality \citep{liu2024lost}. Critically, this context management problem represents a fundamental bottleneck in scaling agent capabilities, as even with modern long-context models, the quadratic attention complexity and information density challenges limit practical deployment.

To address this critical challenge, we reconceptualize context management by drawing an analogy to the attention mechanisms in neural architectures \citep{vaswani2017attention}. Inspired by how Transformer models dynamically weight input tokens to predict the next token, our framework reformulates historical context selection as a Next Step Prediction problem. We conceptualize historical interaction chunks as elements in an external memory store and compute relevance scores for each chunk with respect to the immediate next-step decision \citep{besta2024graphofthoughts,chan2023chateval}. Unlike static summarization or fixed-window approaches, our method dynamically modulates the granularity of historical information—ranging from full detail to mere placeholders—based on its predicted utility for the upcoming action. This paradigm shift enables agents to construct context windows that are both compact and maximally informative, adapting fluidly as task requirements evolve. Crucially, this is achieved by implementing an adaptive pressure mechanism, which automatically adjusts compression intensity as the interaction history grows, thereby enabling the agent to maintain effective reasoning capabilities across extended horizons.
Motivated by these considerations, we propose Predictive Adaptive Context Extraction (PACE), a novel framework for dynamic context management in LLM-based agent systems. PACE operationalizes the Next Step Prediction paradigm through a lightweight, vectorized attention scorer that predicts the relevance of each historical interaction chunk. This score, in turn, guides the dynamic extraction and assembly of a hierarchically structured context that balances detail preservation with computational efficiency. By leveraging multi-granularity representations of history and pressure-adaptive thresholds, PACE constructs contexts that are simultaneously concise and comprehensively informative, enabling effective scaling to thousands of interaction steps.

The contributions of this work are summarized as follows:
\begin{itemize}
\item We introduce PACE, a novel framework that reframes context management as a Next Step Prediction problem, dynamically constructing context through vectorized attention and pressure-adaptive compression based on predicted relevance for the immediate next action.

\item We demonstrate comprehensive improvements across six diverse benchmarks and four models of varying scales, including state-of-the-art web agents. PACE achieves larger gains on complex tasks compared to simpler ones, while maintaining robust performance across languages and domains.

\item We significantly extend agent operational longevity, achieving 4,897 interaction steps in ultra-long-horizon scenarios, representing a 66.2$\times$ improvement over the full-context ReAct baseline and 5.1$\times$ over advanced folding baselines.
\end{itemize}



\section{Related Work}
\subsection{Web Agent Systems}
LLM-based web agents represent a powerful paradigm for automating complex web interactions, spanning tasks from information retrieval to autonomous navigation \citep{hu-etal-2025-os}. Recent advancements have focused on enhancing agent autonomy and adaptability. This includes leveraging self-supervised reinforcement learning for dynamic web environments \citep{qi2025webrl}, developing co-evolving world models for iterative self-improvement \citep{fang2025webevolver}, and specializing agents for high-stakes domains \citep{liu2025llm}.

Despite advancing core reasoning and tool-use capabilities, the efficacy of these agents in long-horizon scenarios is fundamentally constrained by the need for robust context management. Our work, Predictive Adaptive Context Extraction (PACE), directly addresses this foundational bottleneck by providing a dynamic mechanism to sustain agent performance across extended interactions.

\subsection{Context Management in LLM Agents}
Effective context management is pivotal for LLM agents, as decision quality degrades with increasing interaction history. While traditional fixed-window truncation is inadequate for ultra-long-horizon tasks \citep{xu-etal-2024-rethinking}, recent work has formalized context engineering as a critical discipline \citep{mei2025surveycontextengineeringlarge}, categorized memory architectures \citep{zhang2025survey}, and developed benchmarks like MemBench \citep{tan2025membenchcomprehensiveevaluationmemory}. Additional approaches include long-context alignment \citep{bai2024longalign}, training-time data sculpting \citep{lu2024datasculptcraftingdatalandscapes}, and multi-agent collaboration \citep{zhang2024chain}.

Innovative compression mechanisms have emerged, including AgentFold \citep{ye2025agentfoldlonghorizonwebagents}, Context-Folding \citep{sun2025scaling}, hierarchical memory management \citep{hu-etal-2025-hiagent}, attention biasing \citep{zhu2025focus,kang2025aconoptimizingcontextcompression}, and treating memory as action \citep{zhang2025memory,liu2025contexttoolcontextmanagement}. These methods dynamically consolidate past interactions through procedural rules or learned policies. PACE distinguishes itself through a predictive attention mechanism that scores each historical chunk's relevance for the immediate next step, combined with pressure-adaptive compression that fluidly adjusts granularity. This enables efficient scaling to ultra-long-horizon tasks where policy-based methods may falter.


% Optimized Methodology Section for PACE paper

% Revised Methodology Section for PACE paper

\section{Methodology}
\label{sec:methodology} 

In this section, we present the Predictive Adaptive Context Extraction framework for dynamic context management in long-horizon agent tasks, as shown in Figure \ref{fig:experiments}. We formalize the problem setup, detail our system architecture with multi-granularity memory representation, explain the predictive attention mechanism with pressure-adaptive compression, and describe the implementation details.

\subsection{Problem Formulation}

Consider an agent executing a complex task requiring $T$ interaction steps. At each step $t$, the agent observes state $o_t$ and selects action $a_t$. The complete interaction history is:
\begin{equation}
\mathcal{H}_t = \{(a_1, o_1), (a_2, o_2), \ldots, (a_{t-1}, o_{t-1})\}
\end{equation}
Traditional agents condition actions on the full history, i.e., $a_t = \pi(o_t, \mathcal{H}_t)$, but this becomes computationally prohibitive as $t$ grows. Moreover, preserving complete history introduces information overload that can degrade decision quality, particularly as distant interactions dilute attention from critical recent context.

The goal is to construct a compressed context $\mathcal{C}_t$ from $\mathcal{H}_t$ that satisfies $|\mathcal{C}_t| \ll |\mathcal{H}_t|$ while enhancing decision quality by filtering noise and emphasizing relevant information:
\begin{equation}
\text{Performance}(\pi(o_t, \mathcal{C}_t)) \geq \text{Performance}(\pi(o_t, \mathcal{H}_t))
\end{equation}
PACE addresses this as a predictive attention problem, predicting which historical chunks are most relevant for the next action $a_t$ and dynamically adjusting their granularity to construct contexts that are both compact and more informative than unfiltered complete history.


\subsection{System Architecture and Multi-Granularity Memory}

PACE's architecture comprises five coordinated components. The \textbf{Main Agent} is the core decision-maker, receiving the dynamically constructed context $\mathcal{C}_t$ to generate actions using a state-of-the-art LLM. The \textbf{External Memory Store} ($\mathcal{M}$) maintains the complete interaction history: $\mathcal{M}_t = \{\text{Chunk}_1, \ldots, \text{Chunk}_{t-1}\}$. The \textbf{Representation Generator} utilizes a LLM (e.g., Gemini 2.5 Flash-Lite) to asynchronously produce multi-level summaries for each chunk, avoiding blockage of the main agent loop. The \textbf{Attention Scorer} employs a locally-hosted dense retrieval model (e.g., BGE-M3) to predict chunk relevance via semantic similarity. Finally, the \textbf{Context Builder} synthesizes relevance scores with adaptive compression policies to assemble $\mathcal{C}_t$ within token budget constraints.

Each interaction chunk maintains four representations at different granularity levels:
\begin{equation}
\begin{split}
\text{Chunk}_i = \{ & id_i, \, type_i, \, t_i, \, R_{\text{full}}^{(i)}, \, R_{\text{detailed}}^{(i)}, \\
& R_{\text{brief}}^{(i)}, \, R_{\text{ph}}^{(i)}, \, \mathbf{k}_i \}
\end{split}
\end{equation}
where $\mathbf{k}_i$ is the pre-computed key embedding. The representations range from complete content ($R_{\text{full}}$), a comprehensive summary ($R_{\text{detailed}}$), a concise 1-2 sentence summary ($R_{\text{brief}}$), to a minimal placeholder ($R_{\text{ph}}$). These multiple granularity levels enable flexible compression adapting to varying relevance scores.

\subsection{Predictive Attention with Adaptive Compression}
PACE's core innovation treats historical context selection as a next-step prediction problem. Let $N$ denote the number of recent chunks preserved in full detail (we set $N=2$). Given the user's query $Q$ and the most recent $N$ chunks, we construct their concatenation $\mathcal{R}_t = Q \oplus \bigoplus_{j=t-N}^{t-1} R_{\text{full}}^{(j)}$. We encode both the query and historical chunks using an embedding model, which produces dense vectors:
\begin{equation}
\begin{aligned}
\mathbf{q}_t &= \text{Enc}(\text{Truncate}(\mathcal{R}_t, L_{\max})) \\
\mathbf{k}_i &= \text{Enc}(\text{Truncate}(R_{\text{full}}^{(i)}, L_{\max}))
\end{aligned}
\end{equation}
where $L_{\max}$ is the encoder's maximum input length, and $\mathbf{k}_i$ is the key embedding for each historical chunk $\text{Chunk}_i$ where $i \leq t-N-1$. This symmetric truncation strategy ensures both queries and keys fully utilize the encoder's capacity while maintaining balanced semantic representation. Key embeddings are computed once at chunk creation time and cached, enabling efficient retrieval even before detailed summaries are generated asynchronously. The summaries ($R_{\text{detailed}}$, $R_{\text{brief}}$) are used solely for multi-granularity presentation in the final context.

Let $M = t - N - 1$ denote the number of chunks requiring scoring. Raw similarity scores are computed via cosine similarity: $s_i = \cos(\mathbf{q}_t, \mathbf{k}_i)$. We then apply softmax with low temperature $\tau = 0.3$ to sharpen the distribution, accentuating relevant chunks while suppressing irrelevant ones:
\begin{equation}
w_i = \frac{\exp(s_i / \tau)}{\sum_{j=1}^{M} \exp(s_j / \tau)}
\end{equation}
To enable adaptive thresholding, we compute relative weights: $\tilde{w}_i = M \cdot w_i$. A relative weight $\tilde{w}_i > 1$ indicates above-average relevance. With low temperature $\tau$, the softmax becomes highly peaked, concentrating weight on top-ranked chunks while driving most $\tilde{w}_i$ values well below 1, naturally facilitating intensified compression.

\textbf{Pressure-Adaptive Thresholds.} A key PACE feature is adapting compression intensity based on interaction state. We define "compression pressure" $P_t \in [0, 1]$ reflecting both task progression and context budget utilization:
\begin{equation}
P_t = \max\left(\frac{t}{T_{\max}}, \frac{|\mathcal{C}_{t-1}|}{B_{\max}}\right)
\end{equation}
where $T_{\max}$ is the expected maximum task length, $B_{\max}$ is the token budget (set to 128K), and $|\mathcal{C}_{t-1}|$ denotes the previous context's token count. In practice, $T_{\max}$ is set to the 95th percentile of observed task lengths in each benchmark's validation set, providing a robust estimate without requiring precise horizon knowledge. We initialize $|\mathcal{C}_0|$ as the system prompt plus query length, ensuring well-defined initial conditions and avoiding circular dependency.

Base thresholds $(\alpha_0, \beta_0, \gamma_0) = (0.4, 0.8, 1.5)$ are adjusted dynamically: $\alpha_t = \alpha_0 \cdot (1 + \lambda P_t)$, and similarly for $\beta_t$ and $\gamma_t$, where $\lambda = 0.5$ controls adaptation rate. As pressure $P_t$ increases, thresholds rise proportionally, making it progressively harder for chunks to qualify for detailed representations. This pushes more chunks toward brief summaries or placeholders, intensifying compression automatically as context accumulates. The adjusted relative weight $\tilde{w}_i$ determines the representation level:
\begin{equation}
\text{Select}(\text{Chunk}_i) = \begin{cases}
R_{\text{full}}^{(i)} & \text{if } \tilde{w}_i > \gamma_t \\
R_{\text{detailed}}^{(i)} & \text{if } \beta_t < \tilde{w}_i \leq \gamma_t \\
R_{\text{brief}}^{(i)} & \text{if } \alpha_t < \tilde{w}_i \leq \beta_t \\
R_{\text{ph}}^{(i)} & \text{if } \tilde{w}_i \leq \alpha_t
\end{cases}
\end{equation}
The final context is assembled as:
\begin{equation}
\begin{split}
\mathcal{C}_t = \text{Sys} \oplus Q \oplus \left(\bigoplus_{i=1}^{M} \text{Sel}_i\right) 
\oplus \left(\bigoplus_{j=t-N}^{t-1} R_{\text{full}}^{(j)}\right)
\end{split}
\end{equation}
where $\text{Sys}$ denotes the system prompt and $\text{Sel}_i = \text{Select}(\text{Chunk}_i)$.

\subsection{Implementation Details}

\textbf{Glimpse Mechanism.} To mitigate potential scorer errors, PACE includes a glimpse tool allowing the agent to explicitly request full details of any over-compressed chunk. The agent can invoke $\text{glimpse}(i) \rightarrow R_{\text{full}}^{(i)}$ when encountering insufficient information. To maintain bounded context growth, we limit glimpse requests to at most 3 per step, with retrieved content included in the next step's token budget. This provides a crucial fallback mechanism while preserving efficiency.

\textbf{Embedding and Summary Generation.} We deploy BGE-M3 locally for vectorized attention computation, with key embeddings computed once at chunk creation and cached, enabling $O(M)$ dot-product operations. BGE-M3 is a multilingual dense retrieval model that natively supports both English and Chinese, enabling cross-lingual robustness without additional fine-tuning. Summary generation runs in background threads; if unavailable when needed, the system falls back to either $R_{\text{full}}$ or $R_{\text{ph}}$ based on budget constraints. The asynchronous architecture ensures that the main agent loop is never blocked waiting for summary generation, maintaining system responsiveness even under heavy load.

\textbf{Computational Complexity.} At each step $t$, PACE performs $O(M)$ dot-product operations for attention scoring, where $M = t - N - 1$ is the number of historical chunks. The softmax computation and threshold-based selection are also $O(M)$. Context assembly involves simple concatenation operations. The overall per-step complexity is linear in the history length, making PACE highly scalable compared to quadratic attention mechanisms in transformers.

The complete workflow is summarized in Algorithm~\ref{alg:pace}.

\begin{algorithm}[t]
\caption{Predictive Adaptive Context Extraction (Single Step)}
\label{alg:pace}
\begin{algorithmic}[1]
\REQUIRE Memory $\mathcal{M}_{t-1}$, observation $o_t$, query $Q$, recent count $N$
\ENSURE Action $a_t$, updated memory $\mathcal{M}_t$
\STATE $\mathbf{q}_t \leftarrow \text{Enc}(Q \oplus R_{\text{full}}^{(t-N:t-1)})$
\STATE $M \leftarrow t - N - 1$
\STATE Retrieve cached $\{\mathbf{k}_i\}_{i=1}^{M}$ from $\mathcal{M}_{t-1}$
\STATE $s_i \leftarrow \cos(\mathbf{q}_t, \mathbf{k}_i)$ for all $i \in \{1, \ldots, M\}$
\STATE $w_i \leftarrow \text{softmax}(s_i / \tau)$
\STATE $\tilde{w}_i \leftarrow M \cdot w_i$
\STATE $P_t \leftarrow \max(t/T_{\max}, |\mathcal{C}_{t-1}|/B_{\max})$
\STATE $(\alpha_t, \beta_t, \gamma_t) \leftarrow \text{AdaptThresh}(P_t)$
\STATE $R_i^* \leftarrow \text{Select}(\text{Chunk}_i, \tilde{w}_i, \alpha_t, \beta_t, \gamma_t)$ for all $i$
\STATE $\mathcal{C}_t \leftarrow \text{Assemble}(Q, \{R_i^*\}, R_{\text{full}}^{(t-N:t-1)})$
\STATE $a_t \leftarrow \pi(o_t, \mathcal{C}_t)$ \COMMENT{Agent may use glimpse tool}
\STATE $o_{t+1} \leftarrow \text{Env}(a_t)$
\STATE Create $\text{Chunk}_t$ with $\mathbf{k}_t = \text{Enc}(\text{Trunc}(R_{\text{full}}^{(t)}))$
\STATE Enqueue async summary generation for $\text{Chunk}_t$
\STATE $\mathcal{M}_t \leftarrow \mathcal{M}_{t-1} \cup \{\text{Chunk}_t\}$
\RETURN $a_t$, $\mathcal{M}_t$
\end{algorithmic}
\end{algorithm}


\begin{table*}[t]
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{BrowseComp} & \textbf{BrowseComp-ZH} & \textbf{WideSearch} & \textbf{GAIA} & \textbf{xbench-DR} & \textbf{WebWalkerQA} \\
\midrule
\rowcolor{gray!20} \multicolumn{7}{c}{\textit{WebSailor-32B}} \\
ReAct Agent & 7.3 & 20.7 & 43.2 & 46.7 & 58.0 & 52.3 \\
Summary Agent & 10.5 & 25.5 & 47.6 & 53.5 & 63.0 & 57.8 \\
Folding Agent & 11.3 & 26.8 & 49.5 & 55.1 & 65.0 & 60.2 \\
PACE (Ours) & \textbf{13.2} & \textbf{29.3} & \textbf{52.8} & \textbf{59.1} & \textbf{68.0} & \textbf{63.5} \\
\midrule
\rowcolor{gray!20} \multicolumn{7}{c}{\textit{DeepSeek-V3.1-671B}} \\
ReAct Agent & 25.8 & 44.3 & 54.8 & 58.3 & 66.0 & 56.4 \\
Summary Agent & 30.0 & 49.2 & 59.3 & 63.0 & 71.0 & 61.2 \\
Folding Agent & 31.6 & 50.9 & 61.2 & 65.4 & 73.0 & 62.8 \\
PACE (Ours) & \textbf{35.1} & \textbf{54.8} & \textbf{65.7} & \textbf{69.3} & \textbf{74.0} & \textbf{66.5} \\
\midrule
\rowcolor{gray!20} \multicolumn{7}{c}{\textit{tongyi-deepresearch-30B}} \\
ReAct Agent & 38.2 & 41.6 & 52.7 & 64.6 & 69.0 & 66.8 \\
Summary Agent & 43.4 & 46.7 & 57.8 & 70.1 & 75.0 & 72.2 \\
Folding Agent & 44.8 & 47.2 & 60.1 & 72.4 & 77.0 & 74.6 \\
PACE (Ours) & \textbf{47.6} & \textbf{51.2} & \textbf{64.2} & \textbf{74.0} & \textbf{81.0} & \textbf{78.1} \\
\midrule
\rowcolor{gray!20} \multicolumn{7}{c}{\textit{Claude-4-Sonnet}} \\
ReAct Agent & 11.4 & 27.2 & 58.4 & 65.2 & 62.0 & 57.9 \\
Summary Agent & 12.2 & 29.1 & 62.0 & 68.5 & 65.0 & 61.7 \\
Folding Agent & 14.5 & 28.3 & 64.0 & 70.1 & 69.0 & 60.9 \\
PACE (Ours) & \textbf{17.8} & \textbf{32.6} & \textbf{67.0} & \textbf{76.4} & \textbf{72.0} & \textbf{65.8} \\
\bottomrule
\end{tabular}}
\caption{Performance comparison of context management methods across six benchmarks. Results averaged over 3 runs. xbench-DR denotes xbench-DeepResearch. The best scores are \textbf{bolded}.}
\label{tab:main_results}
\end{table*}

\section{Experiments}
\subsection{Experimental Setup}

\textbf{Datasets.} We evaluate PACE on six diverse benchmarks: BrowseComp~\citep{wei2025browsecomp} and its Chinese variant BrowseComp-ZH~\citep{zhou2025browsecompzhbenchmarkingwebbrowsing} for compositional web browsing tasks; WideSearch~\citep{wong2025widesearchbenchmarkingagenticbroad} for broad-scope web exploration; GAIA~\citep{mialon2023gaia} for general AI assistant tasks requiring real-world reasoning; xbench-DeepResearch~\citep{chen2025xbenchtrackingagentsproductivity} for deep research capabilities; and WebWalkerQA~\citep{wu2025webwalkerbenchmarkingllmsweb} for web navigation question answering. These benchmarks span varying interaction lengths and task complexities, making them ideal for evaluating long-horizon context management. Detailed evaluation protocols for each benchmark are provided in Appendix~\ref{appendix:evaluation}.

\textbf{Models.} We evaluate four LLMs across different scales. At the medium scale (30-32B parameters), we select WebSailor-32B~\citep{li2025websailornavigatingsuperhumanreasoning}, an open-source agentic model specialized in complex web navigation, and tongyi-deepresearch-30B~\citep{tongyideepresearchteam2025tongyideepresearchtechnicalreport}, the current sota open-source model on web agent tasks with particular strengths in deep research and agent reasoning. For large-scale models, we select DeepSeek-V3.1-671B~\citep{deepseekai2025deepseekv3technicalreport} and Claude-4-Sonnet~\citep{anthropic2025claude4}, which represent leading large-scale LLMs. Deployment details are described in Appendix~\ref{appendix:environment}.

\textbf{Baselines.} We compare against three context management strategies: (1) \textbf{ReAct Agent}~\citep{yao2022react}, which preserves complete interaction histories; (2) \textbf{Summary Agent}~\citep{wu2025resumunlockinglonghorizonsearch}, which generates cumulative summaries at each step; and (3) \textbf{Folding Agent}~\citep{ye2025agentfoldlonghorizonwebagents,sun2025scaling}, which employs adaptive folding mechanisms for selective compression. To ensure fair comparison, all methods use the same summarization model (Gemini 2.5 Flash-Lite) and token budgets across experiments.

\subsection{Main Results}


Table~\ref{tab:main_results} summarizes our findings, which highlight three key advantages of the PACE framework.

\textbf{PACE Redefines the State-of-the-Art in Context Management.} PACE decisively outperforms the full spectrum of baselines, from naive full-history methods to sophisticated folding agents. This superiority is exemplified on the challenging xbench-DeepResearch benchmark, where tongyi-deepresearch-30B equipped with PACE achieves an 81.0\% success rate---a significant +4.0 percentage point gain over the strong Folding Agent baseline. This establishes our predictive attention mechanism as a more effective and generalizable solution.

\textbf{PACE Unlocks the Latent Potential of Powerful LLMs.} The framework creates a powerful synergy with capable models by removing the context bottleneck that often throttles their performance. This is starkly illustrated with Claude-4-Sonnet on the GAIA benchmark, where its success rate leaps to 76.4\%---a remarkable +6.3 percentage point increase over the Folding Agent. This demonstrates that PACE enables models to operate closer to their true reasoning potential.

\textbf{PACE Demonstrates Exceptional Cross-Lingual and Cross-Domain Robustness.} The framework's generalizability is confirmed by its consistent high performance across diverse settings. It delivers comparable gains on both English (BrowseComp) and Chinese (BrowseComp-ZH) benchmarks and proves effective across the full spectrum of tested domains, from web navigation to deep research. This establishes PACE as a fundamental architectural improvement rather than a narrow, domain-specific solution.


\subsection{Ablation Study}

To understand the contribution of individual components within PACE, we conduct ablation experiments on tongyi-deepresearch-30B by selectively removing key mechanisms: (1) multi-granularity representations, retaining only full content and placeholders; (2) pressure-adaptive thresholds, using fixed values throughout execution; (3) low-temperature softmax, reverting to standard temperature $\tau=1.0$; and (4) the glimpse mechanism, disabling on-demand detail retrieval.

\begin{table}[t]
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{BrowseComp} & \textbf{BrowseComp-ZH} & \textbf{WideSearch} & \textbf{GAIA} \\
\midrule
PACE (Full) & \textbf{47.6} & \textbf{51.2} & \textbf{64.2} & \textbf{74.0} \\
\textsl{w/o Multi-gran.} & 42.3 & 45.5 & 55.9 & 68.5 \\
\textsl{w/o Pressure} & 43.9 & 46.4 & 58.4 & 70.9 \\
\textsl{w/o Softmax} & 43.1 & 46.0 & 56.7 & 69.3 \\
\textsl{w/o Glimpse} & 44.2 & 47.5 & 59.3 & 71.7 \\
\midrule
\rowcolor{gray!10} ReAct Agent & 38.2 & 41.6 & 52.7 & 64.6 \\
\rowcolor{gray!10} Summary Agent & 43.4 & 46.7 & 57.8 & 70.1 \\
\rowcolor{gray!10} Folding Agent & 44.8 & 47.2 & 60.1 & 72.4 \\
\bottomrule
\end{tabular}}
\caption{Ablation study showing the contribution of each PACE component. ReAct Agent, Summary Agent, and Folding Agent baselines are included for reference.}
\label{tab:ablation}
\end{table}


As shown in Table~\ref{tab:ablation}, removing multi-granularity representations causes the most severe performance degradation (5.3--8.3\% across datasets), falling below both Folding Agent and Summary Agent baselines and approaching the performance level of ReAct Agent. This confirms that flexible detail levels are the cornerstone of PACE's effectiveness. Removing the low-temperature softmax also results in substantial degradation (4.7--7.5\%), falling below both Summary Agent and Folding Agent on most benchmarks, which demonstrates that sharpening the attention distribution is critical for accurately identifying relevant historical chunks. The pressure-adaptive mechanism yields moderate performance drops, with the ablated variant performing comparably to Summary Agent but below Folding Agent. Notably, \textsl{w/o Pressure} falls below Summary Agent on BrowseComp-ZH (46.4\% vs 46.7\%), suggesting that adaptive compression becomes particularly important for certain task characteristics. The glimpse mechanism provides the smallest but consistent contribution (2.3--4.9\%), with the ablated variant still outperforming Summary Agent but falling short of Folding Agent on most benchmarks. The full PACE system consistently outperforms all ablated variants and all baselines, demonstrating that these components work synergistically to achieve optimal performance that surpasses even the sophisticated Folding Agent approach.

\subsection{Context Growth Analysis}

To quantify PACE's efficiency in managing context accumulation, we analyze token usage dynamics across interaction steps on GAIA Level 3 tasks, which frequently require extended execution horizons. Figure~\ref{fig:context_growth} compares the context growth patterns of ReAct, Summary Agent, and PACE under a 128K token budget.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{context_growth.png}
\caption{Context token usage over interaction steps on GAIA Level 3 tasks (128K budget).}
\label{fig:context_growth}
\end{figure}



ReAct exhibits near-linear growth as it preserves complete interaction history, rapidly exhausting the 128K budget and forcing termination at step 39. Summary Agent mitigates this through cumulative summarization, extending viable execution to step 131---a $3.4\times$ improvement over ReAct. However, its compression strategy remains fundamentally linear, inevitably reaching the budget ceiling.

In contrast, PACE demonstrates fundamentally different dynamics. During the first 20 steps, context grows approximately linearly, reaching 27.1K tokens. Beyond this point, the pressure-adaptive mechanism effectively bounds growth, with context size exhibiting fluctuations while maintaining a gradual upward trend as the system dynamically balances information retention against budget constraints.


Notably, PACE's context usage at step 200 (45.1K) remains comparable to step 20 (27.1K), representing only a $1.7\times$ increase despite a $10\times$ growth in interaction length. At step 39 (ReAct's termination point), PACE uses 28.5K tokens compared to ReAct's 128K—a 78\% reduction. At step 131 (Summary Agent's termination point), PACE maintains 48.0K tokens versus Summary Agent's 128K—a 63\% reduction. This bounded growth directly enables the performance gains observed in Table~\ref{tab:main_results}: while baselines reach their context limits and are forced to perform crude truncation that permanently discards information from distant interactions, leading to degraded performance, PACE sustains effective reasoning across the full task horizon, translating context efficiency into higher task completion rates.

\subsection{Performance vs.\ Task Complexity}

To assess robustness under increasing task difficulty, we analyze GAIA's three-level difficulty hierarchy and compare success rates across methods. Level 1 tasks are simple problems requiring minimal reasoning and tool use, Level 2 tasks demand moderate multi-step reasoning with increased complexity, and Level 3 tasks involve highly complex long-horizon problems requiring sophisticated planning and multi-tool coordination.
\begin{table}[t]
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}{llccc}
\toprule
\textbf{Model} & \textbf{Method} & \textbf{Level 1} & \textbf{Level 2} & \textbf{Level 3} \\
\midrule
\multirow{4}{*}{\textbf{tongyi-deepresearch-30B}} 
& ReAct Agent & 83.3 & 57.6 & 42.1 \\
& Summary Agent & 85.7 & 65.2 & 47.4 \\
& Folding Agent & 85.7 & 68.2 & 52.6 \\
& PACE (Ours) & \textbf{88.1} & \textbf{68.2} & \textbf{57.9} \\
\midrule
\multirow{4}{*}{\textbf{Claude-4-Sonnet}} 
& ReAct Agent & 83.3 & 59.1 & 42.1 \\
& Summary Agent & 83.3 & 63.6 & 47.4 \\
& Folding Agent & 85.7 & 63.6 & 52.6 \\
& PACE (Ours) & \textbf{88.1} & \textbf{71.2} & \textbf{63.2} \\
\bottomrule
\end{tabular}}
\caption{Success Rate (\%) across GAIA difficulty levels. PACE's advantage grows substantially with task complexity.}
\label{tab:complexity}
\end{table}



As shown in Table~\ref{tab:complexity}, all methods perform comparably on simpler Level 1 tasks where context management is less critical, with differences within 5 percentage points. However, the performance gap widens dramatically with increasing complexity. On Level 2 tasks, PACE with Claude-4-Sonnet achieves 71.2\%, a +12.1 point gain over ReAct (59.1\%) and +7.6 points over Folding Agent (63.6\%). The most striking results emerge on challenging Level 3 tasks: PACE enables Claude-4-Sonnet to reach 63.2\%, representing a +21.1 point improvement over ReAct (42.1\%) and +10.6 points over Folding Agent (52.6\%). This escalating advantage reflects PACE's core strength: as task horizons extend, traditional approaches suffer from either context overflow (ReAct) or information loss (Summary and Folding), while PACE's pressure-adaptive mechanism dynamically balances detail preservation with compression efficiency.



\subsection{Ultra-Long Horizon Stress Test}

To rigorously probe the scaling limits of our framework under realistic constraints, we designed an ultra-long-horizon stress test. Standard benchmarks do not naturally generate thousands of interaction steps, so we constructed a single ``meta-task'' by concatenating all questions from GAIA's Level 2 and Level 3 test sets. The agent, powered by Gemini 2.5 Pro, was instructed to solve every question sequentially within one continuous session. We configured the agent with a 256K token context window, defining failure as the point where this limit was exceeded. This design forces the agent to manage a vast and complex history spanning dozens of distinct sub-tasks.

\begin{figure}[t]
\centering
\includegraphics[width=0.8\linewidth]{ultra_long_horizon_steps.png}
\caption{Operational longevity comparison in the ultra-long-horizon stress test (256K context budget).}
\label{fig:stress_test}
\end{figure}
The results, depicted in Figure~\ref{fig:stress_test}, reveal a stark divergence in operational longevity. The standard ReAct agent reached its context limit in a mere 74 steps, overwhelmed by the history of just a few sub-tasks. The Summary Agent extended this to 233 steps but lost track of the overall task structure due to non-selective compression. The more advanced Folding Agent fared significantly better, reaching 954 steps before its context became unmanageable.

In stark contrast, PACE with its standard adaptive pressure ($\lambda=0.5$) sustained operation for 2,776 steps---a $2.9\times$ improvement in operational longevity over the Folding Agent and a remarkable $37.5\times$ improvement over the full-context ReAct baseline. Even more remarkably, by increasing the adaptation rate for aggressive compression ($\lambda=1.0$), PACE extended its operational horizon to 4,897 steps, achieving a $5.1\times$ improvement over Folding Agent and an exceptional $66.2\times$ improvement over ReAct. This result powerfully demonstrates that PACE's predictive, pressure-adaptive mechanism is critical for navigating ultra-long-horizon tasks, enabling agents to maintain coherent reasoning over thousands of interaction steps within a finite, albeit large, context window.


\section{Conclusion}

We presented PACE, a framework that addresses context management in long-horizon agent tasks by reframing it as a Next Step Prediction problem. PACE dynamically constructs compact, informative contexts using lightweight attention scoring and pressure-adaptive compression. Extensive evaluations demonstrate substantial performance gains across diverse domains and model scales, while extending operational longevity to 4,897 steps—a 66.2$\times$ improvement over the full-context ReAct baseline. This work fundamentally extends the capability of LLM-based agents in ultra-long-horizon scenarios that were previously intractable.