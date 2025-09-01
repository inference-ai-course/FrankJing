#!/usr/bin/env python3
"""
Complete Demo Script for Week 5 Hybrid Retrieval System
======================================================

This script demonstrates the complete workflow:
1. Build hybrid index
2. Run evaluation
3. Start API server
4. Show example usage

Run this script to see the full system in action.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False

def check_week4_data():
    """Check if Week 4 data exists."""
    print("🔍 Checking Week 4 data...")
    
    week4_path = Path("../HomeWork4")
    required_files = ["faiss.index", "embeddings.npy", "chunks.jsonl", "meta.json"]
    
    missing_files = []
    for file in required_files:
        if not (week4_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing Week 4 files: {missing_files}")
        print("Please ensure Week 4 data exists in ../HomeWork4/")
        print("You can build it by running the Week 4 homework first.")
        return False
    else:
        print("✅ All Week 4 files found")
        return True

def main():
    """Run the complete demo."""
    
    print("🚀 Week 5: Hybrid Retrieval System - Complete Demo")
    print("=" * 60)
    
    # Check prerequisites
    if not check_week4_data():
        return
    
    # Step 1: Test the system
    print("\n" + "=" * 60)
    print(" STEP 1: Testing System")
    print("=" * 60)
    
    if not run_command("python test_system.py", "Testing hybrid retrieval system"):
        print("❌ System tests failed. Please fix the issues before continuing.")
        return
    
    # Step 2: Build hybrid index
    print("\n" + "=" * 60)
    print(" STEP 2: Building Hybrid Index")
    print("=" * 60)
    
    if not run_command("python hybrid_search.py build --db hybrid_index.db", "Building hybrid index"):
        print("❌ Failed to build hybrid index")
        return
    
    # Step 3: Run demo
    print("\n" + "=" * 60)
    print(" STEP 3: Running Demo")
    print("=" * 60)
    
    if not run_command("python demo.py", "Running hybrid search demo"):
        print("❌ Demo failed")
        return
    
    # Step 4: Run evaluation
    print("\n" + "=" * 60)
    print(" STEP 4: Running Evaluation")
    print("=" * 60)
    
    if not run_command("python simple_evaluation.py", "Running evaluation"):
        print("❌ Evaluation failed")
        return
    
    # Step 5: Start API server
    print("\n" + "=" * 60)
    print(" STEP 5: Starting API Server")
    print("=" * 60)
    
    print("🌐 Starting FastAPI server...")
    print("The server will start in the background.")
    print("You can access the API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server when you're done.")
    
    # Start the server
    try:
        subprocess.run("python fastapi_hybrid.py", shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    
    # Final summary
    print("\n" + "=" * 60)
    print(" DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n🎉 What you've accomplished:")
    print("✅ Built a hybrid retrieval system combining FAISS and SQLite FTS5")
    print("✅ Implemented weighted sum and RRF fusion strategies")
    print("✅ Created a FastAPI web service for search")
    print("✅ Evaluated system performance with multiple metrics")
    print("✅ Demonstrated the system with example queries")
    
    print("\n📁 Files created:")
    print("• hybrid_index.db - SQLite database with metadata and FTS5 index")
    print("• evaluation_results.json - Performance evaluation results")
    print("• Various Python modules for the hybrid search system")
    
    print("\n🔧 Next steps:")
    print("• Experiment with different alpha values for weighted fusion")
    print("• Try different embedding models")
    print("• Add more test queries to the evaluation")
    print("• Build a web interface for the search system")
    print("• Implement result caching for better performance")
    
    print("\n📚 Key learning outcomes:")
    print("• Understanding of hybrid search architectures")
    print("• Experience with score fusion techniques")
    print("• Knowledge of SQLite FTS5 and FAISS integration")
    print("• Evaluation methodology for search systems")
    print("• API development for search services")

if __name__ == "__main__":
    main()
