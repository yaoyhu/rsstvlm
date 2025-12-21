from typing import Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from rsstvlm.logger import rag_logger

IMPROVED_T2C_PROMPT = PromptTemplate(
    """You are a Neo4j Cypher query expert. Generate a Cypher query to answer the user's question.

Database Schema:
{schema}

IMPORTANT GUIDELINES:
1. **Use flexible matching**: Use CONTAINS, STARTS WITH, or case-insensitive matching instead of exact matches
2. **Search broadly**: Don't assume exact entity names - use patterns to find related entities
3. **Return rich context**: Include entity properties, relationship types, and connected nodes
4. **Handle synonyms**: Consider that concepts may be represented with different terms
5. **Limit results**: Always add LIMIT clause (typically 10-50 results)
6. **Return useful data**: Return entity names, properties, and relationship information

QUERY PATTERNS TO USE:

Pattern 1 - Fuzzy Entity Search:
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower('keyword')
RETURN e.name, labels(e), e LIMIT 20

Pattern 2 - Find Related Entities:
MATCH (e1:Entity)-[r]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS toLower('keyword1')
  OR toLower(e1.name) CONTAINS toLower('keyword2')
RETURN e1.name, type(r), e2.name, e1, e2
LIMIT 30

Pattern 3 - Path Finding:
MATCH path = (e1:Entity)-[*1..2]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS toLower('keyword1')
  AND toLower(e2.name) CONTAINS toLower('keyword2')
RETURN e1.name, [r in relationships(path) | type(r)], e2.name
LIMIT 20

Pattern 4 - Relationship Type Search:
MATCH (e1:Entity)-[r:RELATIONSHIP_TYPE]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS toLower('keyword')
RETURN e1.name, type(r), e2.name, properties(r)
LIMIT 20

Pattern 5 - Property-based Search:
MATCH (e:Entity)
WHERE any(prop IN keys(e) WHERE toLower(toString(e[prop])) CONTAINS toLower('keyword'))
RETURN e.name, properties(e)
LIMIT 20

EXAMPLES:

Question: "What is the relationship between O3 and pollution?"
Good Query:
MATCH (e1:Entity)-[r]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS 'o3'
  AND (toLower(e2.name) CONTAINS 'pollut' 
       OR toLower(type(r)) CONTAINS 'pollut'
       OR any(prop IN keys(e2) WHERE toLower(toString(e2[prop])) CONTAINS 'pollut'))
RETURN e1.name, type(r) AS relationship, e2.name, properties(e2)
LIMIT 30

Question: "How does temperature affect O3?"
Good Query:
MATCH (e1:Entity)-[r]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS 'temperature'
  AND toLower(e2.name) CONTAINS 'o3'
RETURN e1.name, type(r), e2.name, properties(r)
UNION
MATCH (e1:Entity)-[r]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS 'o3'
  AND (toLower(type(r)) CONTAINS 'temperature' 
       OR toLower(type(r)) CONTAINS 'affect'
       OR toLower(type(r)) CONTAINS 'impact')
RETURN e1.name, type(r), e2.name, properties(r)
LIMIT 30

Question: "What factors influence ozone formation?"
Good Query:
MATCH (factor:Entity)-[r]->(o3:Entity)
WHERE toLower(o3.name) CONTAINS 'o3' 
   OR toLower(o3.name) CONTAINS 'ozone'
RETURN factor.name, type(r) AS influence_type, o3.name
UNION
MATCH (o3:Entity)<-[r:RESULTS_FROM|INCREASES_DUE_TO|INFLUENCED_BY]-(factor:Entity)
WHERE toLower(o3.name) CONTAINS 'o3'
   OR toLower(o3.name) CONTAINS 'ozone'
RETURN factor.name, type(r), o3.name
LIMIT 30

BAD PRACTICES (DO NOT DO):
❌ MATCH (e:Entity {{name: 'pollution'}})  // Too specific - entity might not exist
❌ WHERE e.name = 'O3'  // Case-sensitive exact match
❌ No LIMIT clause  // May return too many results
❌ Returning only IDs without context  // Not human-readable

User Question: {query_str}

Generate ONLY the raw Cypher query without any markdown formatting, explanations, or code blocks.
Do NOT include ```cypher or ``` markers.
Just return the pure Cypher query that can be executed directly.

Cypher Query:"""
)


SIMPLE_T2C_PROMPT = PromptTemplate(
    """Generate a Neo4j Cypher query for the following question.

Schema:
{schema}

RULES:
- Use CONTAINS for flexible matching: WHERE toLower(e.name) CONTAINS toLower('keyword')
- Always include LIMIT (10-50)
- Return entity names, relationship types, and properties
- Use case-insensitive matching with toLower()
- Consider synonyms and variations of terms

TEMPLATE:
```cypher
MATCH (e1:Entity)-[r]-(e2:Entity)
WHERE toLower(e1.name) CONTAINS toLower('term1')
  AND (toLower(e2.name) CONTAINS toLower('term2')
       OR toLower(type(r)) CONTAINS toLower('term2'))
RETURN e1.name, type(r), e2.name, properties(e2)
LIMIT 20
```

Question: {query_str}

Generate ONLY the raw Cypher query without any markdown formatting, explanations, or code blocks.
Do NOT include ```cypher or ``` markers.
Just return the pure Cypher query that can be executed directly."""
)


class Text2CypherRetriever(BaseRetriever):
    """Custom Text-to-Cypher retriever for Neo4j Property Graph."""

    def __init__(
        self,
        graph_store: Neo4jPropertyGraphStore,
        llm: Any,
        prompt: PromptTemplate | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self._graph_store = graph_store
        self._llm = llm
        self._prompt = prompt or SIMPLE_T2C_PROMPT
        self._verbose = verbose

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Generate Cypher query and retrieve results from Neo4j."""
        query_str = query_bundle.query_str
        print(f"\n🔍 Text2CypherRetriever._retrieve called with: {query_str}")

        # Get schema (use a shorter version to save tokens)
        try:
            schema = self._graph_store.get_schema_str()
            # Truncate schema if too long
            if len(schema) > 8000:
                schema = schema[:8000] + "\n... (schema truncated)"
            if self._verbose:
                print(f"📋 Schema length: {len(schema)} chars")
        except Exception as e:
            print(f"❌ Failed to get schema: {e}")
            import traceback

            traceback.print_exc()
            return []

        # Generate Cypher query using LLM
        try:
            prompt_text = self._prompt.format(
                schema=schema, query_str=query_str
            )

            response = self._llm.complete(prompt_text)
            cypher = response.text

            if self._verbose:
                rag_logger.info(f"Generated Cypher Query:\n{cypher}")
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            import traceback

            traceback.print_exc()
            return []

        # Execute Cypher query
        try:
            results = self._graph_store.structured_query(cypher)
            if self._verbose:
                print(f"✅ Query returned {len(results)} results")
                for i, r in enumerate(results[:5]):
                    print(f"  Result {i + 1}: {r}")
                rag_logger.info(f"Query returned {len(results)} results")
        except Exception as e:
            print(f"❌ Cypher query execution failed: {e}")
            rag_logger.error(f"Cypher query failed: {e}")
            import traceback

            traceback.print_exc()
            return []

        # Convert results to NodeWithScore
        nodes = []
        for i, record in enumerate(results):
            text_parts = []
            for key, value in record.items():
                if isinstance(value, dict):
                    text_parts.append(f"{key}:")
                    for prop_key, prop_value in value.items():
                        if prop_value and prop_key not in (
                            "embedding",
                            "id",
                            "_node_content",
                        ):
                            text_parts.append(f"  {prop_key}: {prop_value}")
                elif value is not None:
                    text_parts.append(f"{key}: {value}")

            text = "\n".join(text_parts)
            if text.strip():
                node = TextNode(
                    text=text,
                    metadata={"source": "neo4j_cypher", "query": cypher},
                )
                nodes.append(NodeWithScore(node=node, score=1.0 - (i * 0.01)))

        print(f"📊 Returning {len(nodes)} nodes")
        return nodes
