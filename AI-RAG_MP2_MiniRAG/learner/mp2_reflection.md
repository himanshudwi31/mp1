# MP2 Reflection

## What worked
(One specific thing that worked — a chunking choice, a prompt detail, etc.)

*Chunking choice was a main differentiator in my approach. With the implementation of Section based chunking, I was able to generate more relevant responses. Facts matched was also improved as the context within chunks was intact and retrieval was more contextual*

## What didn't work
(One specific failure mode you hit. Did chunks get too small? Did retrieval
pull the wrong section? Did the LLM hallucinate?)

*I started with Sliding Window chunking approach as it was easy to implement but it didn't work as the context was split into different chunks due to which generated answer and facts matching were impacted. So I changed the approach and used section based chunking which improved the results and also helped in attaching relevant section headers*

## What I'd change
(One thing you'd do differently with another 5 hours.)

*If I get additional time to further improve the results, I could try different approaches to improve the retrieval context by levearging techniques like Query rewriting, Query Decomposition through LLM and Hybrid retrieval using both Dense (Semantic) and Sparse (Keyword based)search methods.*

## One surprise
(Anything from the build that genuinely surprised you. Could be technical —
"text-embedding-3-small was way better than I expected on rare names" —
or pedagogical — "I didn't realise how much hybrid retrieval was helping
until I tried dense-only here.")

*With the implementation of RAG approach where context is provided to LLM along with user question, I was surprised to see relevant responses without writing complex system and user prompt. We need not be very elaborate and detailed in instructions to LLM which saves time, reduces iterations while maintaining the quality in output.*
