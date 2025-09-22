"""
Human annotation interface for collecting preference data.
Provides both command-line and web-based interfaces for annotating summary pairs.
"""

import json
import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import webbrowser
import tempfile
from datetime import datetime

from utils import setup_logging, format_timestamp, save_jsonl, load_jsonl

class AnnotationInterface:
    """Base class for annotation interfaces."""
    
    def __init__(self):
        self.logger = setup_logging("INFO")
        self.annotations_dir = Path("data/annotations")
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
    def annotate_summary_pairs(self, summary_pairs: List[Dict[str, Any]], 
                             annotator_id: str = "human") -> List[Dict[str, Any]]:
        """Annotate a list of summary pairs."""
        raise NotImplementedError("Subclasses must implement annotate_summary_pairs")
    
    def save_annotations(self, annotations: List[Dict[str, Any]], 
                        filepath: Optional[str] = None) -> str:
        """Save annotations to file."""
        if filepath is None:
            timestamp = format_timestamp()
            filepath = self.annotations_dir / f"annotations_{timestamp}.jsonl"
        
        save_jsonl(annotations, str(filepath))
        self.logger.info(f"Saved {len(annotations)} annotations to {filepath}")
        return str(filepath)
    
    def load_annotations(self, filepath: str) -> List[Dict[str, Any]]:
        """Load annotations from file."""
        return load_jsonl(filepath)

class CommandLineAnnotationInterface(AnnotationInterface):
    """Command-line interface for annotation."""
    
    def __init__(self):
        super().__init__()
        self.logger.info("Initialized command-line annotation interface")
    
    def annotate_summary_pairs(self, summary_pairs: List[Dict[str, Any]], 
                             annotator_id: str = "human") -> List[Dict[str, Any]]:
        """Annotate summary pairs using command-line interface."""
        annotations = []
        
        print(f"\n{'='*60}")
        print(f"ANNOTATION INTERFACE - {len(summary_pairs)} pairs to annotate")
        print(f"Annotator: {annotator_id}")
        print(f"{'='*60}\n")
        
        for i, pair in enumerate(summary_pairs):
            print(f"\n--- Pair {i+1}/{len(summary_pairs)} ---")
            print(f"Paper: {pair.get('paper_title', 'Unknown')}")
            print(f"Paper ID: {pair['paper_id']}")
            
            # Display summaries
            summary1 = pair['summary_1']
            summary2 = pair['summary_2']
            
            print(f"\n{'='*40}")
            print("SUMMARY 1:")
            print(f"Strategy: {summary1.get('strategy', 'unknown')}")
            print(f"Length: {summary1.get('length', 0)} characters")
            print("-" * 40)
            print(summary1['text'])
            
            print(f"\n{'='*40}")
            print("SUMMARY 2:")
            print(f"Strategy: {summary2.get('strategy', 'unknown')}")
            print(f"Length: {summary2.get('length', 0)} characters")
            print("-" * 40)
            print(summary2['text'])
            
            # Get user preference
            while True:
                try:
                    choice = input(f"\nWhich summary is better? (1/2/tie/skip): ").strip().lower()
                    
                    if choice in ['1', '2', 'tie', 'skip']:
                        break
                    else:
                        print("Please enter 1, 2, tie, or skip")
                        
                except KeyboardInterrupt:
                    print("\nAnnotation interrupted by user")
                    return annotations
                except EOFError:
                    print("\nAnnotation completed")
                    return annotations
            
            # Record annotation
            if choice != 'skip':
                annotation = self._create_annotation(pair, choice, annotator_id)
                annotations.append(annotation)
                
                print(f"✓ Recorded preference: {choice}")
            else:
                print("⏭ Skipped this pair")
        
        print(f"\n{'='*60}")
        print(f"ANNOTATION COMPLETE")
        print(f"Total annotations: {len(annotations)}")
        print(f"{'='*60}\n")
        
        return annotations
    
    def _create_annotation(self, pair: Dict[str, Any], choice: str, 
                          annotator_id: str) -> Dict[str, Any]:
        """Create annotation record."""
        if choice == 'tie':
            preferred_index = 0  # No preference
        elif choice == '1':
            preferred_index = 1
        elif choice == '2':
            preferred_index = 2
        else:
            preferred_index = 0
        
        return {
            'pair_id': pair['pair_id'],
            'paper_id': pair['paper_id'],
            'paper_title': pair.get('paper_title', ''),
            'summary_1': pair['summary_1'],
            'summary_2': pair['summary_2'],
            'preferred_index': preferred_index,
            'choice': choice,
            'annotator_id': annotator_id,
            'timestamp': format_timestamp(),
            'annotation_type': 'preference'
        }

class WebAnnotationInterface(AnnotationInterface):
    """Web-based interface for annotation using HTML/JavaScript."""
    
    def __init__(self):
        super().__init__()
        self.logger.info("Initialized web-based annotation interface")
    
    def annotate_summary_pairs(self, summary_pairs: List[Dict[str, Any]], 
                             annotator_id: str = "human") -> List[Dict[str, Any]]:
        """Annotate summary pairs using web interface."""
        # Create HTML interface
        html_content = self._create_html_interface(summary_pairs, annotator_id)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_file = f.name
        
        # Open in browser
        print(f"Opening annotation interface in browser...")
        print(f"File: {temp_file}")
        webbrowser.open(f"file://{temp_file}")
        
        # Wait for user to complete annotation
        input("\nPress Enter after completing annotations in the browser...")
        
        # Load results (this would need to be implemented with a proper web server)
        # For now, we'll create a simple file-based approach
        results_file = self.annotations_dir / f"web_annotations_{annotator_id}_{format_timestamp()}.json"
        
        print(f"Please save your annotations to: {results_file}")
        input("Press Enter after saving annotations...")
        
        # Load annotations
        try:
            with open(results_file, 'r') as f:
                annotations = json.load(f)
            return annotations
        except FileNotFoundError:
            self.logger.warning("No annotations file found")
            return []
    
    def _create_html_interface(self, summary_pairs: List[Dict[str, Any]], 
                              annotator_id: str) -> str:
        """Create HTML interface for annotation."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary Annotation Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .pair-header {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .summary {
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            background-color: #fafafa;
        }
        .summary-1 { border-left: 4px solid #2196f3; }
        .summary-2 { border-left: 4px solid #ff9800; }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background-color: #4caf50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 0 10px;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .choice-buttons {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
        }
        .choice-btn {
            padding: 15px 30px;
            font-size: 18px;
            border-radius: 8px;
            cursor: pointer;
            border: 2px solid transparent;
        }
        .choice-btn:hover {
            border-color: #333;
        }
        .summary-1-btn {
            background-color: #e3f2fd;
            color: #1976d2;
        }
        .summary-2-btn {
            background-color: #fff3e0;
            color: #f57c00;
        }
        .tie-btn {
            background-color: #f3e5f5;
            color: #7b1fa2;
        }
        .progress {
            background-color: #e0e0e0;
            border-radius: 10px;
            padding: 3px;
            margin: 20px 0;
        }
        .progress-bar {
            background-color: #4caf50;
            height: 20px;
            border-radius: 8px;
            transition: width 0.3s;
        }
        .hidden { display: none; }
        .metadata {
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Summary Annotation Interface</h1>
        <p>Annotator: {annotator_id}</p>
        <div class="progress">
            <div class="progress-bar" id="progressBar" style="width: 0%"></div>
        </div>
        <p id="progressText">Pair 1 of {len(summary_pairs)}</p>
    </div>

    <div id="annotationContainer">
        {annotation_forms}
    </div>

    <div class="container">
        <div class="controls">
            <button onclick="exportResults()">Export Annotations</button>
            <button onclick="downloadResults()">Download Results</button>
        </div>
    </div>

    <script>
        let currentPair = 0;
        let annotations = [];
        const totalPairs = {len(summary_pairs)};

        function showPair(index) {{
            // Hide all pairs
            document.querySelectorAll('.pair-container').forEach(el => {{
                el.classList.add('hidden');
            }});
            
            // Show current pair
            document.getElementById(`pair-${{index}}`).classList.remove('hidden');
            
            // Update progress
            const progress = ((index + 1) / totalPairs) * 100;
            document.getElementById('progressBar').style.width = progress + '%';
            document.getElementById('progressText').textContent = `Pair ${{index + 1}} of ${{totalPairs}}`;
        }}

        function selectChoice(pairIndex, choice) {{
            const pair = {summary_pairs_json}[pairIndex];
            const annotation = {{
                pair_id: pair.pair_id,
                paper_id: pair.paper_id,
                paper_title: pair.paper_title,
                summary_1: pair.summary_1,
                summary_2: pair.summary_2,
                preferred_index: choice === 'tie' ? 0 : (choice === '1' ? 1 : 2),
                choice: choice,
                annotator_id: '{annotator_id}',
                timestamp: new Date().toISOString(),
                annotation_type: 'preference'
            }};
            
            annotations[pairIndex] = annotation;
            
            // Move to next pair
            if (currentPair < totalPairs - 1) {{
                currentPair++;
                showPair(currentPair);
            }} else {{
                alert('All pairs annotated! You can now export the results.');
            }}
        }}

        function exportResults() {{
            const results = annotations.filter(a => a !== undefined);
            console.log('Annotations:', results);
            alert(`Exported ${{results.length}} annotations. Check console for details.`);
        }}

        function downloadResults() {{
            const results = annotations.filter(a => a !== undefined);
            const dataStr = JSON.stringify(results, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'annotations.json';
            link.click();
        }}

        // Initialize
        showPair(0);
    </script>
</body>
</html>
        """
        
        # Create annotation forms for each pair
        annotation_forms = ""
        for i, pair in enumerate(summary_pairs):
            form = f"""
            <div class="pair-container" id="pair-{i}" {'class="hidden"' if i > 0 else ''}>
                <div class="container">
                    <div class="pair-header">
                        <h2>Pair {i+1}: {pair.get('paper_title', 'Unknown Paper')}</h2>
                        <p>Paper ID: {pair['paper_id']}</p>
                    </div>
                    
                    <div class="summary summary-1">
                        <h3>Summary 1</h3>
                        <div class="metadata">
                            Strategy: {pair['summary_1'].get('strategy', 'unknown')} | 
                            Length: {pair['summary_1'].get('length', 0)} characters
                        </div>
                        <p>{pair['summary_1']['text']}</p>
                    </div>
                    
                    <div class="summary summary-2">
                        <h3>Summary 2</h3>
                        <div class="metadata">
                            Strategy: {pair['summary_2'].get('strategy', 'unknown')} | 
                            Length: {pair['summary_2'].get('length', 0)} characters
                        </div>
                        <p>{pair['summary_2']['text']}</p>
                    </div>
                    
                    <div class="choice-buttons">
                        <button class="choice-btn summary-1-btn" onclick="selectChoice({i}, '1')">
                            Choose Summary 1
                        </button>
                        <button class="choice-btn summary-2-btn" onclick="selectChoice({i}, '2')">
                            Choose Summary 2
                        </button>
                        <button class="choice-btn tie-btn" onclick="selectChoice({i}, 'tie')">
                            Tie
                        </button>
                    </div>
                </div>
            </div>
            """
            annotation_forms += form
        
        # Convert summary pairs to JSON for JavaScript
        import json
        summary_pairs_json = json.dumps(summary_pairs)
        
        return html_template.format(
            annotator_id=annotator_id,
            len=len(summary_pairs),
            annotation_forms=annotation_forms,
            summary_pairs_json=summary_pairs_json
        )

class BatchAnnotationInterface(AnnotationInterface):
    """Batch annotation interface for processing multiple annotators."""
    
    def __init__(self):
        super().__init__()
        self.logger.info("Initialized batch annotation interface")
    
    def create_annotation_batches(self, summary_pairs: List[Dict[str, Any]], 
                                 batch_size: int = 10) -> List[List[Dict[str, Any]]]:
        """Split summary pairs into batches for annotation."""
        batches = []
        for i in range(0, len(summary_pairs), batch_size):
            batch = summary_pairs[i:i + batch_size]
            batches.append(batch)
        
        self.logger.info(f"Created {len(batches)} annotation batches of size {batch_size}")
        return batches
    
    def merge_annotations(self, annotation_files: List[str]) -> List[Dict[str, Any]]:
        """Merge annotations from multiple files."""
        all_annotations = []
        
        for file_path in annotation_files:
            annotations = self.load_annotations(file_path)
            all_annotations.extend(annotations)
        
        self.logger.info(f"Merged {len(all_annotations)} annotations from {len(annotation_files)} files")
        return all_annotations
    
    # create_reward_training_data method moved to data_formatter.py for better separation of concerns
