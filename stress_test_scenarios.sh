#!/bin/bash

# Stress Testing Scenarios Script
# This script helps run different load test scenarios as part of the MLOps assignment

echo "=== IRIS API Stress Testing Scenarios ==="

# Function to run basic load test with 1000 concurrent connections
run_basic_load_test() {
    echo "Running basic load test with 1000 concurrent connections..."
    echo "Command: wrk -t4 -c1000 -d30s http://$EXTERNAL_IP/predict"
    wrk -t4 -c1000 -d30s http://$EXTERNAL_IP/predict
}

# Function to run high concurrency test with 2000 concurrent connections
run_high_concurrency_test() {
    echo "Running high concurrency test with 2000 concurrent connections..."
    echo "Command: wrk -t4 -c2000 -d30s http://$EXTERNAL_IP/predict"
    wrk -t4 -c2000 -d30s http://$EXTERNAL_IP/predict
}

# Function to check HPA status
check_hpa_status() {
    echo "Checking Horizontal Pod Autoscaler status..."
    kubectl get hpa
    echo ""
    echo "Checking pod status..."
    kubectl get pods
}

# Function to demonstrate scaling behavior
demo_scaling_behavior() {
    echo "=== Demonstrating Scaling Behavior ==="
    echo "1. Check current HPA configuration:"
    kubectl get hpa iris-api-hpa -o yaml
    echo ""
    echo "2. Check current pod status before load test:"
    kubectl get pods
    echo ""
    echo "3. Running load test that should trigger scaling..."
    run_basic_load_test
    echo ""
    echo "4. Checking pod status after load test:"
    kubectl get pods
}

# Function to demonstrate constrained scaling (maxReplicas: 1)
demo_constrained_scaling() {
    echo "=== Demonstrating Constrained Scaling (maxReplicas: 1) ==="
    echo "Setting HPA with maxReplicas: 1..."
    
    # Apply HPA configuration with maxReplicas: 1
    cat > temp-hpa.yaml << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iris-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iris-api
  minReplicas: 1
  maxReplicas: 1
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
EOF
    
    kubectl apply -f temp-hpa.yaml
    echo ""
    echo "Running load test with constrained scaling (2000 connections)..."
    run_high_concurrency_test
    echo ""
    echo "Checking pod status after constrained load test:"
    kubectl get pods
    echo ""
    echo "Checking HPA status:"
    kubectl get hpa
    
    # Reset to original HPA configuration
    kubectl delete -f temp-hpa.yaml
    rm temp-hpa.yaml
}

# Main execution logic
if [ $# -eq 0 ]; then
    echo "Usage: $0 [scenario]"
    echo "Scenarios:"
    echo "  basic            - Run basic load test (1000 connections)"
    echo "  high_concurrency - Run high concurrency test (2000 connections)" 
    echo "  check_hpa        - Check HPA and pod status"
    echo "  demo_scaling     - Demonstrate scaling behavior"
    echo "  demo_constrained - Demonstrate constrained scaling (maxReplicas: 1)"
    exit 1
fi

SCENARIO=$1

case $SCENARIO in
    basic)
        run_basic_load_test
        ;;
    high_concurrency)
        run_high_concurrency_test
        ;;
    check_hpa)
        check_hpa_status
        ;;
    demo_scaling)
        demo_scaling_behavior
        ;;
    demo_constrained)
        demo_constrained_scaling
        ;;
    *)
        echo "Unknown scenario: $SCENARIO"
        exit 1
        ;;
esac

echo "=== Stress testing completed ==="