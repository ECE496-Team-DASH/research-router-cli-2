#!/usr/bin/env python3

import sys
import traceback

def test_imports():
    """Test all the imports needed for GraphRAG functionality."""
    
    try:
        print("Testing nano-graphrag import...")
        import nano_graphrag
        print("✓ nano-graphrag imported successfully")
        
        print("\nTesting nano-graphrag components...")
        from nano_graphrag import GraphRAG, QueryParam
        print("✓ GraphRAG and QueryParam imported successfully")
        
        print("\nTesting elia_chat imports...")
        from elia_chat.graphrag_manager import is_graphrag_available, GraphRAGManager
        print("✓ GraphRAG manager imports successful")
        
        print("\nTesting availability check...")
        available = is_graphrag_available()
        print(f"✓ GraphRAG available: {available}")
        
        print("\nTesting document parsing imports...")
        from elia_chat.graphrag_manager import DocumentParser
        print("✓ DocumentParser imported successfully")
        
        print("\nTesting PDF support...")
        try:
            import PyPDF2
            print("✓ PyPDF2 available for PDF processing")
        except ImportError:
            print("⚠ PyPDF2 not available (this is expected)")
        
        print("\nTesting DOCX support...")
        try:
            import docx
            print("✓ python-docx available for DOCX processing")
        except ImportError:
            print("⚠ python-docx not available (this is expected)")
        
        print("\nAll tests passed! GraphRAG integration should work.")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
