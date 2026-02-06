#!/bin/bash
# =============================================================================
# Ray Hang Debugging Script
# =============================================================================
# 이 스크립트는 Ray 클러스터에서 hang이 발생했을 때 디버깅 정보를 수집합니다.
# =============================================================================

set -e

echo "=============================================="
echo "Ray Hang Debugging Information"
echo "=============================================="
echo ""

# 1. Ray 클러스터 상태 확인
echo "1. Ray Cluster Status:"
echo "----------------------------------------------"
ray status 2>&1 || echo "Ray status failed - Ray may not be running"
echo ""

# 2. Ray Dashboard URL 확인
echo "2. Ray Dashboard URL:"
echo "----------------------------------------------"
RAY_DASHBOARD=$(ray status 2>/dev/null | grep -oP 'dashboard at \K[^\s]+' || echo "Not found")
if [ "$RAY_DASHBOARD" != "Not found" ]; then
    echo "Dashboard URL: http://$RAY_DASHBOARD"
    echo "  - Overview: http://$RAY_DASHBOARD/#/overview"
    echo "  - Actors: http://$RAY_DASHBOARD/#/actors"
    echo "  - Tasks: http://$RAY_DASHBOARD/#/tasks"
    echo "  - Logs: http://$RAY_DASHBOARD/#/logs"
else
    echo "Dashboard URL not found"
fi
echo ""

# 3. Ray Actors 상태 확인
echo "3. Ray Actors Status:"
echo "----------------------------------------------"
python3 << 'EOF'
import ray
import sys

try:
    ray.init(address='auto', ignore_reinit_error=True)
    
    # 모든 actors 가져오기
    actors = ray.util.list_actors()
    
    if not actors:
        print("No actors found")
    else:
        print(f"Total actors: {len(actors)}")
        print("")
        
        for actor in actors:
            print(f"Actor ID: {actor['actor_id']}")
            print(f"  Name: {actor.get('name', 'N/A')}")
            print(f"  State: {actor.get('state', 'N/A')}")
            print(f"  Class: {actor.get('class_name', 'N/A')}")
            print(f"  PID: {actor.get('pid', 'N/A')}")
            print(f"  Node ID: {actor.get('node_id', 'N/A')}")
            
            # Pending tasks 확인
            if 'num_pending_tasks' in actor:
                print(f"  Pending Tasks: {actor['num_pending_tasks']}")
            
            # Recent failures 확인
            if 'num_restarts' in actor:
                print(f"  Restarts: {actor['num_restarts']}")
            
            print("")
            
except Exception as e:
    print(f"Error getting actor status: {e}")
    sys.exit(1)
EOF
echo ""

# 4. Ray Tasks 상태 확인 (pending/running)
echo "4. Ray Tasks Status:"
echo "----------------------------------------------"
python3 << 'EOF'
import ray
import sys
from collections import defaultdict

try:
    ray.init(address='auto', ignore_reinit_error=True)
    
    # 모든 tasks 가져오기
    tasks = ray.util.list_tasks()
    
    if not tasks:
        print("No tasks found")
    else:
        # State별로 그룹화
        state_counts = defaultdict(int)
        pending_tasks = []
        running_tasks = []
        
        for task in tasks:
            state = task.get('state', 'UNKNOWN')
            state_counts[state] += 1
            
            if state == 'PENDING_ARGS_READY' or state == 'PENDING_NODE_ASSIGNMENT':
                pending_tasks.append(task)
            elif state == 'RUNNING':
                running_tasks.append(task)
        
        print(f"Total tasks: {len(tasks)}")
        print("Tasks by state:")
        for state, count in sorted(state_counts.items()):
            print(f"  {state}: {count}")
        print("")
        
        if pending_tasks:
            print(f"Pending tasks ({len(pending_tasks)}):")
            for task in pending_tasks[:10]:  # 최대 10개만 표시
                print(f"  Task ID: {task.get('task_id', 'N/A')}")
                print(f"    Name: {task.get('name', 'N/A')}")
                print(f"    State: {task.get('state', 'N/A')}")
                print(f"    Actor ID: {task.get('actor_id', 'N/A')}")
                print("")
        
        if running_tasks:
            print(f"Running tasks ({len(running_tasks)}):")
            for task in running_tasks[:10]:  # 최대 10개만 표시
                print(f"  Task ID: {task.get('task_id', 'N/A')}")
                print(f"    Name: {task.get('name', 'N/A')}")
                print(f"    State: {task.get('state', 'N/A')}")
                print(f"    Actor ID: {task.get('actor_id', 'N/A')}")
                print(f"    Worker PID: {task.get('worker_pid', 'N/A')}")
                print("")
                
except Exception as e:
    print(f"Error getting task status: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
echo ""

# 5. Ray Workers 상태 확인
echo "5. Ray Workers Status:"
echo "----------------------------------------------"
python3 << 'EOF'
import ray
import sys

try:
    ray.init(address='auto', ignore_reinit_error=True)
    
    # 모든 workers 가져오기
    nodes = ray.nodes()
    
    if not nodes:
        print("No nodes found")
    else:
        print(f"Total nodes: {len(nodes)}")
        print("")
        
        for node in nodes:
            node_id = node.get('NodeID', 'N/A')
            is_alive = node.get('Alive', False)
            resources = node.get('Resources', {})
            
            print(f"Node ID: {node_id}")
            print(f"  Alive: {is_alive}")
            print(f"  Resources: {resources}")
            print("")
            
except Exception as e:
    print(f"Error getting worker status: {e}")
    sys.exit(1)
EOF
echo ""

# 6. 프로세스 상태 확인 (Ray 관련)
echo "6. Ray-related Processes:"
echo "----------------------------------------------"
echo "Ray processes:"
ps aux | grep -E "ray|vllm|python.*verl" | grep -v grep | head -20 || echo "No Ray processes found"
echo ""

# 7. GPU 사용 상태 확인
echo "7. GPU Usage:"
echo "----------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,processes --format=csv,noheader,nounits | head -10
else
    echo "nvidia-smi not available"
fi
echo ""

# 8. Ray 로그 위치 안내
echo "8. Ray Logs Location:"
echo "----------------------------------------------"
RAY_LOG_DIR="${HOME}/.ray/logs"
if [ -d "$RAY_LOG_DIR" ]; then
    echo "Ray logs directory: $RAY_LOG_DIR"
    echo "Recent log files:"
    find "$RAY_LOG_DIR" -name "*.log" -type f -mtime -1 | head -10 || echo "No recent logs found"
else
    echo "Ray logs directory not found at $RAY_LOG_DIR"
    echo "Try: find ~ -name 'ray-*' -type d 2>/dev/null | head -5"
fi
echo ""

# 9. Python 스택 트레이스 확인 (가능한 경우)
echo "9. Python Stack Traces (if available):"
echo "----------------------------------------------"
python3 << 'EOF'
import ray
import sys

try:
    ray.init(address='auto', ignore_reinit_error=True)
    
    # Ray의 내부 상태 확인
    print("Ray cluster resources:")
    print(ray.available_resources())
    print("")
    
    # 현재 실행 중인 작업 확인
    print("Ray internal state (limited):")
    # Ray의 내부 API는 버전에 따라 다를 수 있음
    try:
        from ray import _private
        print("Ray private module available")
    except:
        print("Ray private module not accessible")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
echo ""

echo "=============================================="
echo "Debugging Tips:"
echo "=============================================="
echo "1. Ray Dashboard에서 Actors/Tasks 탭 확인"
echo "2. Pending 상태의 tasks 확인"
echo "3. 특정 actor의 로그 확인: ray logs <actor_id>"
echo "4. 프로세스 스택 트레이스: kill -USR1 <pid> (Python 프로세스)"
echo "5. Ray 클러스터 재시작: ray stop && ray start --head"
echo ""
