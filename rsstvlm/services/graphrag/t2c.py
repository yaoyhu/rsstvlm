from llama_index.core.prompts.base import (
    PromptTemplate,
    PromptType,
)
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from rsstvlm.utils import deepseek

LONG_NEBULAGRAPH_NL2CYPHER_PROMPT_TMPL_STR = """
Create a **NebulaGraph flavor Cypher query** based on provided schema and a question.

The query should be able to try best answer the question with the given graph schema.

NebulaGraph flavor Cypher differs from standard Cypher in the following ways:

- Fully qualify property references with the node's label.

```
// Standard Cypher(incorrect in NebulaGraph)
MATCH (p:person)-[:follow]->() RETURN p.name
// NebulaGraph Cypher, here we use p.person.name with Label specified
MATCH (p:person)-[:follow]->() RETURN p.person.name
```

- Use == for equality comparisons instead of =

```
// Standard Cypher(WRONG)
MATCH (p:person {{name: 'Alice'}}) RETURN p
// NebulaGraph flavor Cypher
MATCH (p:person) WHERE p.person.name == 'Alice' RETURN p
```

With these differentiations, construct a NebulaGraph Cypher query to answer the given question, only return the plain text query, no explanation, apologies, or other text.

NOTE:
0. Try to get as much graph data as possible to answer the question
1. Use valid NebulaGraph flavor Cypher syntax when referring vertex property like `... WHERE v.person.name == 'Alice'...`
2. Adhering strictly to the relationships and properties given in the schema

---
Question: {query_str}
---
Schema: {schema}
---

NebulaGraph flavor Query:
"""

T2C_PROMPT = PromptTemplate(
    LONG_NEBULAGRAPH_NL2CYPHER_PROMPT_TMPL_STR,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

# t2c_qe = KnowledgeGraphQueryEngine(
#     storage_context=self.storage_context,
#     graph_query_synthesis_prompt=T2C_PROMPT,
#     llm=deepseek,
#     verbose=True,
# )
