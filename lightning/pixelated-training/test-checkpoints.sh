#!/bin/bash
# Automated testing script for Wayfarer checkpoint comparison
# Integrates with the evaluation framework we built in Option A

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
TEST_SCENARIOS="${TEST_SCENARIOS:-10}"
REPETITIONS="${REPETITIONS:-3}"
OUTPUT_DIR="test_results"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

# Test scenarios for checkpoint evaluation
declare -a TEST_CASES=(
    '{"message": "I feel anxious about my job interview tomorrow", "scenario": "anxiety", "difficulty": "beginner"}'
    '{"message": "I have been feeling worthless and questioning my existence", "scenario": "depression", "difficulty": "intermediate"}'
    '{"message": "It is hard for me to talk about what happened to me", "scenario": "trauma", "difficulty": "advanced"}'
    '{"message": "I lost my mother last month and feel overwhelmed with grief", "scenario": "grief", "difficulty": "beginner"}'
    '{"message": "My partner and I keep fighting and I do not know what to do", "scenario": "relationship", "difficulty": "intermediate"}'
)

# Function to test a specific checkpoint
test_checkpoint() {
    local checkpoint_name=$1
    local api_url=$2
    local test_case=$3
    local rep=$4
    
    log "Testing $checkpoint_name (rep $rep)"
    
    # Make API request
    start_time=$(date +%s.%N)
    response=$(curl -s -X POST "$api_url/chat" \
        -H "Content-Type: application/json" \
        -d "$test_case" \
        --max-time 30)
    end_time=$(date +%s.%N)
    
    # Calculate response time
    response_time=$(echo "$end_time - $start_time" | bc -l)
    
    # Extract response content
    response_content=$(echo "$response" | jq -r '.response // empty')
    generation_time=$(echo "$response" | jq -r '.generation_time // 0')
    tokens_generated=$(echo "$response" | jq -r '.tokens_generated // 0')
    
    # Log results
    echo "$checkpoint_name,$rep,$response_time,$generation_time,$tokens_generated,\"$response_content\"" >> "$OUTPUT_DIR/raw_results.csv"
    
    echo "$response_content"
}

# Function to run comprehensive checkpoint tests
run_checkpoint_tests() {
    log "🧪 Starting comprehensive checkpoint testing"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Initialize results file
    echo "checkpoint,repetition,api_response_time,generation_time,tokens_generated,response_content" > "$OUTPUT_DIR/raw_results.csv"
    
    # Test each scenario against each checkpoint
    for i in "${!TEST_CASES[@]}"; do
        test_case="${TEST_CASES[$i]}"
        scenario_name="scenario_$((i+1))"
        
        log "📋 Testing $scenario_name: $(echo "$test_case" | jq -r '.message')"
        
        # Test primary checkpoint (300)
        for rep in $(seq 1 $REPETITIONS); do
            test_checkpoint "checkpoint-300" "$API_BASE_URL" "$test_case" "$rep"
            sleep 2  # Brief pause between requests
        done
        
        log "✅ Completed $scenario_name testing"
    done
    
    log "✅ All checkpoint tests completed"
}

# Function to analyze results
analyze_results() {
    log "📊 Analyzing test results"
    
    # Create analysis script
    cat > "$OUTPUT_DIR/analyze.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_checkpoint_performance():
    # Load results
    df = pd.read_csv('raw_results.csv')
    
    # Calculate summary statistics
    summary = df.groupby('checkpoint').agg({
        'api_response_time': ['mean', 'std', 'min', 'max'],
        'generation_time': ['mean', 'std', 'min', 'max'],
        'tokens_generated': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    # Save summary
    summary.to_csv('performance_summary.csv')
    
    # Calculate response quality metrics (simple heuristics)
    df['response_length'] = df['response_content'].str.len()
    df['empathy_score'] = df['response_content'].str.count(r'feel|understand|support|help', case=False)
    df['therapeutic_words'] = df['response_content'].str.count(r'therapy|therapeutic|emotion|anxiety|depression', case=False)
    
    # Quality summary
    quality_summary = df.groupby('checkpoint').agg({
        'response_length': ['mean', 'std'],
        'empathy_score': ['mean', 'std'],
        'therapeutic_words': ['mean', 'std']
    }).round(3)
    
    quality_summary.to_csv('quality_summary.csv')
    
    # Create recommendations
    recommendations = []
    
    # Performance recommendations
    fastest_checkpoint = summary['api_response_time']['mean'].idxmin()
    recommendations.append(f"Fastest Response: {fastest_checkpoint}")
    
    most_tokens = summary['tokens_generated']['mean'].idxmax()
    recommendations.append(f"Most Detailed Responses: {most_tokens}")
    
    # Quality recommendations
    most_empathetic = quality_summary['empathy_score']['mean'].idxmax()
    recommendations.append(f"Most Empathetic: {most_empathetic}")
    
    most_therapeutic = quality_summary['therapeutic_words']['mean'].idxmax()
    recommendations.append(f"Most Therapeutic Language: {most_therapeutic}")
    
    # Save recommendations
    with open('recommendations.txt', 'w') as f:
        f.write("Wayfarer Checkpoint Performance Analysis\n")
        f.write("=" * 40 + "\n\n")
        for rec in recommendations:
            f.write(f"• {rec}\n")
        
        f.write(f"\nDetailed Analysis:\n")
        f.write(f"Performance Summary:\n{summary.to_string()}\n\n")
        f.write(f"Quality Summary:\n{quality_summary.to_string()}\n")
    
    print("Analysis complete!")
    print("\nQuick Recommendations:")
    for rec in recommendations:
        print(f"  • {rec}")

if __name__ == "__main__":
    analyze_checkpoint_performance()
EOF

    # Run analysis
    cd "$OUTPUT_DIR"
    python3 analyze.py
    cd ..
    
    log "📊 Analysis complete - check $OUTPUT_DIR/recommendations.txt"
}

# Function to generate HTML report
generate_report() {
    log "📋 Generating HTML report"
    
    cat > "$OUTPUT_DIR/report.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Wayfarer Checkpoint Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; color: #2c3e50; }
        .summary { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .recommendations { background: #e8f5e8; padding: 20px; border-radius: 5px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #e3f2fd; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Wayfarer Checkpoint Performance Report</h1>
        <p>Generated on: <span id="timestamp"></span></p>
    </div>
    
    <div class="summary">
        <h2>📊 Test Summary</h2>
        <div class="metric">
            <strong>Scenarios Tested:</strong> 5
        </div>
        <div class="metric">
            <strong>Repetitions:</strong> 3 per scenario
        </div>
        <div class="metric">
            <strong>Total API Calls:</strong> 15
        </div>
    </div>
    
    <div class="recommendations">
        <h2>🎯 Key Recommendations</h2>
        <p><em>Detailed recommendations available in recommendations.txt</em></p>
        <ul>
            <li>✅ <strong>Production Deployment:</strong> Use checkpoint that balances speed and quality</li>
            <li>🚀 <strong>Speed Critical:</strong> Use fastest checkpoint for real-time applications</li>
            <li>🎭 <strong>Quality Critical:</strong> Use most empathetic checkpoint for sensitive scenarios</li>
        </ul>
    </div>
    
    <h2>📈 Performance Data</h2>
    <p>Raw performance data available in: <code>performance_summary.csv</code></p>
    <p>Quality metrics available in: <code>quality_summary.csv</code></p>
    <p>Complete raw data available in: <code>raw_results.csv</code></p>
    
    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
    </script>
</body>
</html>
EOF

    log "📋 HTML report generated: $OUTPUT_DIR/report.html"
}

# Main execution
main() {
    log "🚀 Starting Wayfarer checkpoint testing"
    
    # Check if API is available
    if ! curl -s -f "$API_BASE_URL/health" > /dev/null; then
        echo "❌ API not available at $API_BASE_URL"
        echo "   Make sure Wayfarer is running: ./deploy.sh"
        exit 1
    fi
    
    # Run tests
    run_checkpoint_tests
    
    # Analyze results
    if command -v python3 &> /dev/null && python3 -c "import pandas" &> /dev/null; then
        analyze_results
    else
        log "⚠️  Python3 with pandas not available - skipping detailed analysis"
    fi
    
    # Generate report
    generate_report
    
    log "✅ Checkpoint testing complete!"
    log "📁 Results saved in: $OUTPUT_DIR/"
    log "🌐 View report: open $OUTPUT_DIR/report.html"
    
    # Display quick summary
    echo ""
    echo "Quick Summary:"
    echo "• Raw results: $OUTPUT_DIR/raw_results.csv"
    echo "• HTML report: $OUTPUT_DIR/report.html"
    echo "• Recommendations: $OUTPUT_DIR/recommendations.txt"
}

# Run the script
main "$@"