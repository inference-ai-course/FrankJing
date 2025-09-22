"""
Simple test script to verify the pipeline components work correctly.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils import setup_logging, format_timestamp
        print("✓ utils imported successfully")
        
        from data_collector import PaperCollector, DataFormatter
        print("✓ data_collector imported successfully")
        
        from summary_generator import SummaryGenerator, MultimodalSummaryGenerator
        print("✓ summary_generator imported successfully")
        
        from annotation_interface import CommandLineAnnotationInterface, WebAnnotationInterface, BatchAnnotationInterface
        print("✓ annotation_interface imported successfully")
        
        from reward_trainer import RewardModelTrainer, RewardModelEvaluator
        print("✓ reward_trainer imported successfully")
        
        from evaluator import SummaryEvaluator, ComparativeAnalyzer
        print("✓ evaluator imported successfully")
        
        from main import MultimodalSummarizationPipeline
        print("✓ main imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_initialization():
    """Test that components can be initialized."""
    print("\nTesting component initialization...")
    
    try:
        from utils import setup_logging
        logger = setup_logging("INFO")
        print("✓ Logger initialized")
        
        from data_collector import PaperCollector
        collector = PaperCollector()
        print("✓ PaperCollector initialized")
        
        from data_collector import DataFormatter
        formatter = DataFormatter()
        print("✓ DataFormatter initialized")
        
        from annotation_interface import CommandLineAnnotationInterface
        annotation_interface = CommandLineAnnotationInterface()
        print("✓ AnnotationInterface initialized")
        
        from evaluator import SummaryEvaluator
        evaluator = SummaryEvaluator()
        print("✓ SummaryEvaluator initialized")
        
        from evaluator import ComparativeAnalyzer
        analyzer = ComparativeAnalyzer()
        print("✓ ComparativeAnalyzer initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        traceback.print_exc()
        return False

def test_directory_creation():
    """Test that directories are created correctly."""
    print("\nTesting directory creation...")
    
    try:
        from main import MultimodalSummarizationPipeline
        pipeline = MultimodalSummarizationPipeline()
        
        # Check if directories exist
        directories = [
            "data/papers",
            "data/summaries", 
            "data/annotations",
            "outputs",
            "models",
            "logs"
        ]
        
        for directory in directories:
            if Path(directory).exists():
                print(f"✓ Directory {directory} exists")
            else:
                print(f"✗ Directory {directory} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Directory creation failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without heavy operations."""
    print("\nTesting basic functionality...")
    
    try:
        from utils import clean_text, truncate_text, format_timestamp
        
        # Test text cleaning
        test_text = "This is a test text with   extra   spaces."
        cleaned = clean_text(test_text)
        assert cleaned == "This is a test text with extra spaces."
        print("✓ Text cleaning works")
        
        # Test text truncation
        long_text = "This is a very long text that should be truncated because it exceeds the maximum length limit."
        truncated = truncate_text(long_text, 20)
        assert len(truncated) <= 20
        print("✓ Text truncation works")
        
        # Test timestamp formatting
        timestamp = format_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
        print("✓ Timestamp formatting works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MULTIMODAL SUMMARIZATION PIPELINE - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization),
        ("Directory Creation Test", test_directory_creation),
        ("Basic Functionality Test", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nTo run the pipeline:")
        print("python main.py --mode full --num-papers 5 --num-eval-papers 5")
        print("\nFor more options:")
        print("python main.py --help")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

