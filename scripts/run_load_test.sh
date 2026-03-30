#!/bin/bash

# Load test script for AgriPrice Prophet
echo "🚀 AgriPrice Prophet Load Testing Script"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configurations
USERS=(100 500 1000)
CONTAINERS=(1 3 5)
DURATION="3m"

# Create results directory
mkdir -p locust/results

echo -e "${YELLOW}Testing with different container counts...${NC}"

for containers in "${CONTAINERS[@]}"; do
    echo -e "\n${GREEN}Testing with $containers container(s)${NC}"
    
    # Scale the API
    docker-compose up --scale api=$containers -d
    
    # Wait for containers to be ready
    echo "Waiting for containers to be ready..."
    sleep 15
    
    # Run load test for each user count
    for users in "${USERS[@]}"; do
        echo -e "  Testing with $users users..."
        
        # Run locust in headless mode
        locust -f locust/locustfile.py \
            --headless \
            -u $users \
            -r 10 \
            --run-time $DURATION \
            --host=http://localhost:8000 \
            --csv=locust/results/containers_${containers}_users_${users} \
            --html=locust/results/report_${containers}_${users}.html \
            --only-summary
            
        echo -e "  ${GREEN}✓ Completed${NC}"
        sleep 5
    done
    
    # Stop containers
    docker-compose down
    sleep 5
done

echo -e "\n${GREEN}All load tests completed!${NC}"
echo "Results saved in locust/results/"

# Generate summary report
echo -e "\n${YELLOW}Generating summary report...${NC}"

cat > locust/results/summary.md << EOF
# Load Test Summary

## Test Configuration
- Duration: $DURATION per test
- Spawn rate: 10 users/second
- Tool: Locust

## Results

| Containers | Users | Avg Response Time | RPS | Error Rate |
|------------|-------|-------------------|-----|------------|
EOF

# Parse results and add to summary
for containers in "${CONTAINERS[@]}"; do
    for users in "${USERS[@]}"; do
        csv_file="locust/results/containers_${containers}_users_${users}_stats.csv"
        if [ -f "$csv_file" ]; then
            # Extract metrics
            avg_response=$(tail -n 1 "$csv_file" | cut -d',' -f6 | xargs)
            rps=$(tail -n 1 "$csv_file" | cut -d',' -f10 | xargs)
            errors=$(tail -n 1 "$csv_file" | cut -d',' -f13 | xargs)
            
            echo "| $containers | $users | ${avg_response}ms | $rps | ${errors}% |" >> locust/results/summary.md
        fi
    done
done

echo -e "${GREEN}Summary report created: locust/results/summary.md${NC}"
echo -e "\nDone!"