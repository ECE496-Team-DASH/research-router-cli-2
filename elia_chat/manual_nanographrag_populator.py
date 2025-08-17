"""
Manual Nano-GraphRAG Storage Populator

Creates and populates nano-graphrag storage files manually without waiting for nano-graphrag insert.
Supports direct manipulation of GraphML entities/relationships, JSON key-value stores, and vector databases.
"""

import json
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re


class NanoGraphRAGPopulator:
    """Manual populator for nano-graphrag storage files."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.graphml_path = self.storage_dir / "graph_chunk_entity_relation.graphml"
        self.community_reports_path = self.storage_dir / "kv_store_community_reports.json"
        self.full_docs_path = self.storage_dir / "kv_store_full_docs.json"
        self.llm_cache_path = self.storage_dir / "kv_store_llm_response_cache.json"
        self.text_chunks_path = self.storage_dir / "kv_store_text_chunks.json"
        self.entities_path = self.storage_dir / "vdb_entities.json"
        
        # Initialize storage structures
        self.entities = {}
        self.relationships = {}
        self.text_chunks = {}
        self.full_docs = {}
        self.community_reports = {}
        self.llm_cache = {}
        self.entities_vdb = []
        
    def generate_id(self, content: str) -> str:
        """Generate consistent ID from content."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_document(self, doc_id: str, content: str, title: str = None) -> None:
        """Add a full document to storage."""
        self.full_docs[doc_id] = {
            "content": content,
            "title": title or f"Document {doc_id}",
            "metadata": {
                "doc_id": doc_id,
                "length": len(content),
                "created": "manual_insert"
            }
        }
    
    def add_text_chunk(self, chunk_id: str, content: str, doc_id: str, 
                      chunk_order: int = 0, tokens: int = None) -> None:
        """Add a text chunk to storage."""
        self.text_chunks[chunk_id] = {
            "content": content,
            "tokens": tokens or len(content.split()),
            "chunk_order_index": chunk_order,
            "full_doc_id": doc_id
        }
    
    def add_entity(self, entity_name: str, entity_type: str, description: str, 
                  source_id: str) -> str:
        """Add an entity to the graph."""
        entity_id = self.generate_id(f"{entity_name}_{entity_type}")
        
        self.entities[entity_id] = {
            "name": entity_name,
            "type": entity_type,
            "description": description,
            "source_id": source_id
        }
        
        # Add to vector database format
        self.entities_vdb.append({
            "id": entity_id,
            "name": entity_name,
            "type": entity_type,
            "description": description,
            "embedding": [0.0] * 384  # Placeholder embedding
        })
        
        return entity_id
    
    def add_relationship(self, source_entity: str, target_entity: str, 
                        relation_type: str, description: str, weight: float = 1.0,
                        source_id: str = None) -> str:
        """Add a relationship between entities."""
        rel_id = self.generate_id(f"{source_entity}_{relation_type}_{target_entity}")
        
        self.relationships[rel_id] = {
            "source": source_entity,
            "target": target_entity,
            "type": relation_type,
            "description": description,
            "weight": weight,
            "source_id": source_id or "manual_insert"
        }
        
        return rel_id
    
    def add_community_report(self, community_id: str, title: str, summary: str,
                           findings: List[str], rating: float = 7.5) -> None:
        """Add a community analysis report."""
        self.community_reports[community_id] = {
            "id": community_id,
            "title": title,
            "summary": summary,
            "full_content": summary,
            "rank": rating,
            "rank_explanation": f"Community rated {rating}/10 based on analysis",
            "findings": findings,
            "rating": rating,
            "rating_explanation": f"Rated {rating} for relevance and comprehensiveness"
        }
    
    def extract_entities_from_text(self, text: str, doc_id: str) -> List[Tuple[str, str, str]]:
        """Simple entity extraction from text."""
        # Basic named entity patterns
        entities = []
        
        # Find capitalized words (potential proper nouns)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for word in words:
            if len(word) > 2 and word not in ['The', 'This', 'That', 'When', 'Where']:
                entity_type = "PERSON" if word.istitle() else "ENTITY"
                description = f"Entity '{word}' found in document {doc_id}"
                entities.append((word, entity_type, description))
        
        return entities
    
    def process_document_auto(self, doc_id: str, content: str, title: str = None,
                            chunk_size: int = 1200) -> None:
        """Automatically process a document and extract entities/relationships."""
        # Add full document
        self.add_document(doc_id, content, title)
        
        # Split into chunks
        words = content.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        for i, chunk_content in enumerate(chunks):
            chunk_id = f"chunk-{self.generate_id(chunk_content)}"
            self.add_text_chunk(chunk_id, chunk_content, doc_id, i)
            
            # Extract entities from chunk
            entities = self.extract_entities_from_text(chunk_content, chunk_id)
            
            prev_entity_id = None
            for entity_name, entity_type, description in entities:
                entity_id = self.add_entity(entity_name, entity_type, description, chunk_id)
                
                # Create relationship with previous entity in same chunk
                if prev_entity_id:
                    self.add_relationship(
                        prev_entity_id, entity_id, "CO_OCCURS",
                        f"Entities appear together in {chunk_id}", 1.0, chunk_id
                    )
                prev_entity_id = entity_id
        
        # Add basic community report
        self.add_community_report(
            f"community-{doc_id}", 
            f"Analysis of {title or doc_id}",
            f"Community formed around document {doc_id} containing various entities and relationships.",
            [f"Document contains {len(chunks)} text chunks", 
             f"Extracted {len(self.entities)} entities"]
        )
    
    def save_graphml(self) -> None:
        """Save entities and relationships to GraphML format."""
        # Create XML structure
        graphml = ET.Element("graphml")
        graphml.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
        
        # Define attributes
        node_type = ET.SubElement(graphml, "key")
        node_type.set("id", "entity_type")
        node_type.set("for", "node")
        node_type.set("attr.name", "entity_type")
        node_type.set("attr.type", "string")
        
        node_desc = ET.SubElement(graphml, "key")
        node_desc.set("id", "description")
        node_desc.set("for", "node")
        node_desc.set("attr.name", "description")
        node_desc.set("attr.type", "string")
        
        node_source = ET.SubElement(graphml, "key")
        node_source.set("id", "source_id")
        node_source.set("for", "node")
        node_source.set("attr.name", "source_id")
        node_source.set("attr.type", "string")
        
        edge_weight = ET.SubElement(graphml, "key")
        edge_weight.set("id", "weight")
        edge_weight.set("for", "edge")
        edge_weight.set("attr.name", "weight")
        edge_weight.set("attr.type", "double")
        
        edge_desc = ET.SubElement(graphml, "key")
        edge_desc.set("id", "description")
        edge_desc.set("for", "edge")
        edge_desc.set("attr.name", "description")
        edge_desc.set("attr.type", "string")
        
        edge_source = ET.SubElement(graphml, "key")
        edge_source.set("id", "source_id")
        edge_source.set("for", "edge")
        edge_source.set("attr.name", "source_id")
        edge_source.set("attr.type", "string")
        
        # Create graph
        graph = ET.SubElement(graphml, "graph")
        graph.set("id", "G")
        graph.set("edgedefault", "undirected")
        
        # Add nodes (entities)
        for entity_id, entity_data in self.entities.items():
            node = ET.SubElement(graph, "node")
            node.set("id", entity_id)
            
            # Add entity type
            data_type = ET.SubElement(node, "data")
            data_type.set("key", "entity_type")
            data_type.text = entity_data["type"]
            
            # Add description
            data_desc = ET.SubElement(node, "data")
            data_desc.set("key", "description")
            data_desc.text = entity_data["description"]
            
            # Add source
            data_source = ET.SubElement(node, "data")
            data_source.set("key", "source_id")
            data_source.text = entity_data["source_id"]
        
        # Add edges (relationships)
        for rel_id, rel_data in self.relationships.items():
            edge = ET.SubElement(graph, "edge")
            edge.set("id", rel_id)
            edge.set("source", rel_data["source"])
            edge.set("target", rel_data["target"])
            
            # Add weight
            data_weight = ET.SubElement(edge, "data")
            data_weight.set("key", "weight")
            data_weight.text = str(rel_data["weight"])
            
            # Add description
            data_desc = ET.SubElement(edge, "data")
            data_desc.set("key", "description")
            data_desc.text = rel_data["description"]
            
            # Add source
            data_source = ET.SubElement(edge, "data")
            data_source.set("key", "source_id")
            data_source.text = rel_data["source_id"]
        
        # Pretty print and save
        rough_string = ET.tostring(graphml, 'unicode')
        reparsed = minidom.parseString(rough_string)
        
        with open(self.graphml_path, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="  "))
    
    def save_all_files(self) -> None:
        """Save all storage files."""
        # Save GraphML
        self.save_graphml()
        
        # Save JSON files
        with open(self.full_docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.full_docs, f, indent=2, ensure_ascii=False)
        
        with open(self.text_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.text_chunks, f, indent=2, ensure_ascii=False)
        
        with open(self.community_reports_path, 'w', encoding='utf-8') as f:
            json.dump(self.community_reports, f, indent=2, ensure_ascii=False)
        
        with open(self.llm_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.llm_cache, f, indent=2, ensure_ascii=False)
        
        with open(self.entities_path, 'w', encoding='utf-8') as f:
            json.dump(self.entities_vdb, f, indent=2, ensure_ascii=False)
    
    def load_existing_files(self) -> None:
        """Load existing storage files if they exist."""
        # Load JSON files
        for file_path, storage_dict in [
            (self.full_docs_path, self.full_docs),
            (self.text_chunks_path, self.text_chunks),
            (self.community_reports_path, self.community_reports),
            (self.llm_cache_path, self.llm_cache)
        ]:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    storage_dict.update(json.load(f))
        
        # Load entities vector database
        if self.entities_path.exists():
            with open(self.entities_path, 'r', encoding='utf-8') as f:
                self.entities_vdb = json.load(f)
        
        # Load GraphML (basic parsing)
        if self.graphml_path.exists():
            try:
                tree = ET.parse(self.graphml_path)
                root = tree.getroot()
                
                # Parse nodes
                for node in root.findall(".//{http://graphml.graphdrawing.org/xmlns}node"):
                    node_id = node.get("id")
                    entity_data = {"name": node_id, "type": "UNKNOWN", "description": "", "source_id": ""}
                    
                    for data in node.findall(".//{http://graphml.graphdrawing.org/xmlns}data"):
                        key = data.get("key")
                        if key == "entity_type":
                            entity_data["type"] = data.text or ""
                        elif key == "description":
                            entity_data["description"] = data.text or ""
                        elif key == "source_id":
                            entity_data["source_id"] = data.text or ""
                    
                    self.entities[node_id] = entity_data
                
                # Parse edges
                for edge in root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge"):
                    edge_id = edge.get("id")
                    rel_data = {
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "type": "UNKNOWN",
                        "description": "",
                        "weight": 1.0,
                        "source_id": ""
                    }
                    
                    for data in edge.findall(".//{http://graphml.graphdrawing.org/xmlns}data"):
                        key = data.get("key")
                        if key == "weight":
                            rel_data["weight"] = float(data.text or 1.0)
                        elif key == "description":
                            rel_data["description"] = data.text or ""
                        elif key == "source_id":
                            rel_data["source_id"] = data.text or ""
                    
                    self.relationships[edge_id] = rel_data
                    
            except Exception as e:
                print(f"Warning: Could not parse existing GraphML file: {e}")


def create_manual_populator(storage_dir: str) -> NanoGraphRAGPopulator:
    """Create a manual populator instance."""
    return NanoGraphRAGPopulator(Path(storage_dir))


# Example usage
if __name__ == "__main__":
    # Example of manual population
    populator = create_manual_populator("./test_nanographrag_storage")
    
    # Load existing files if any
    populator.load_existing_files()
    
    # Add a document manually
    sample_doc = """
    This is a sample research paper about artificial intelligence and machine learning.
    The authors John Smith and Jane Doe discuss various algorithms including neural networks
    and deep learning approaches. The research was conducted at MIT and Stanford University.
    """
    
    populator.process_document_auto("doc-sample-001", sample_doc, "AI Research Paper")
    
    # Add custom entities
    mit_id = populator.add_entity("MIT", "ORGANIZATION", "Massachusetts Institute of Technology", "doc-sample-001")
    stanford_id = populator.add_entity("Stanford University", "ORGANIZATION", "Stanford University", "doc-sample-001")
    
    # Add relationship
    populator.add_relationship(mit_id, stanford_id, "COLLABORATES_WITH", 
                             "Both institutions collaborate on AI research", 8.5, "doc-sample-001")
    
    # Save all files
    populator.save_all_files()
    
    print(f"Created nano-graphrag storage with:")
    print(f"- {len(populator.entities)} entities")
    print(f"- {len(populator.relationships)} relationships") 
    print(f"- {len(populator.text_chunks)} text chunks")
    print(f"- {len(populator.full_docs)} documents")
    print(f"- {len(populator.community_reports)} community reports")