#!/usr/bin/env python3
"""
Ray Pending Tasks/Actors Checker
이 스크립트는 Ray 클러스터에서 pending 상태의 tasks나 actors를 찾아서 보여줍니다.
"""
import ray
import sys
from collections import defaultdict

def main():
    try:
        # Ray 클러스터에 연결
        ray.init(address='auto', ignore_reinit_error=True)
        
        print("=" * 60)
        print("Ray Hang Debugging - Pending Tasks/Actors")
        print("=" * 60)
        print()
        
        # 1. Actors 상태 확인
        print("1. Actors Status:")
        print("-" * 60)
        try:
            actors = ray.util.list_actors()
            
            if not actors:
                print("No actors found")
            else:
                # State별로 그룹화
                actor_states = defaultdict(list)
                for actor in actors:
                    state = actor.get('state', 'UNKNOWN')
                    actor_states[state].append(actor)
                
                print(f"Total actors: {len(actors)}")
                for state, actor_list in sorted(actor_states.items()):
                    print(f"  {state}: {len(actor_list)}")
                
                # Pending이나 문제가 있는 actors 상세 정보
                problem_states = ['PENDING_CREATION', 'DEAD']
                for state in problem_states:
                    if state in actor_states:
                        print(f"\n{state} Actors:")
                        for actor in actor_states[state][:10]:  # 최대 10개
                            print(f"  Actor ID: {actor.get('actor_id', 'N/A')}")
                            print(f"    Name: {actor.get('name', 'N/A')}")
                            print(f"    Class: {actor.get('class_name', 'N/A')}")
                            print(f"    PID: {actor.get('pid', 'N/A')}")
                            print(f"    Node ID: {actor.get('node_id', 'N/A')}")
                            if 'num_pending_tasks' in actor:
                                print(f"    Pending Tasks: {actor['num_pending_tasks']}")
                            print()
        except Exception as e:
            print(f"Error getting actors: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        
        # 2. Tasks 상태 확인
        print("2. Tasks Status:")
        print("-" * 60)
        try:
            tasks = ray.util.list_tasks()
            
            if not tasks:
                print("No tasks found")
            else:
                # State별로 그룹화
                task_states = defaultdict(list)
                for task in tasks:
                    state = task.get('state', 'UNKNOWN')
                    task_states[state].append(task)
                
                print(f"Total tasks: {len(tasks)}")
                for state, task_list in sorted(task_states.items()):
                    print(f"  {state}: {len(task_list)}")
                
                # Pending tasks 상세 정보
                pending_states = ['PENDING_ARGS_READY', 'PENDING_NODE_ASSIGNMENT', 'PENDING_OBJ_STORE_MEM_AVAILABLE']
                has_pending = False
                for state in pending_states:
                    if state in task_states:
                        has_pending = True
                        print(f"\n{state} Tasks:")
                        for task in task_states[state][:20]:  # 최대 20개
                            print(f"  Task ID: {task.get('task_id', 'N/A')}")
                            print(f"    Name: {task.get('name', 'N/A')}")
                            print(f"    Actor ID: {task.get('actor_id', 'N/A')}")
                            print(f"    Function: {task.get('func_or_class_name', 'N/A')}")
                            if 'start_time_ms' in task:
                                print(f"    Start Time: {task['start_time_ms']} ms")
                            print()
                
                if not has_pending:
                    print("\nNo pending tasks found")
                
                # Running tasks 확인 (오래 실행 중인 것들)
                if 'RUNNING' in task_states:
                    print(f"\nRunning Tasks ({len(task_states['RUNNING'])}):")
                    for task in task_states['RUNNING'][:10]:  # 최대 10개
                        print(f"  Task ID: {task.get('task_id', 'N/A')}")
                        print(f"    Name: {task.get('name', 'N/A')}")
                        print(f"    Actor ID: {task.get('actor_id', 'N/A')}")
                        print(f"    Worker PID: {task.get('worker_pid', 'N/A')}")
                        if 'start_time_ms' in task:
                            print(f"    Start Time: {task['start_time_ms']} ms")
                        print()
                        
        except Exception as e:
            print(f"Error getting tasks: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        
        # 3. Ray 클러스터 리소스 상태
        print("3. Ray Cluster Resources:")
        print("-" * 60)
        try:
            resources = ray.available_resources()
            print("Available resources:")
            for key, value in sorted(resources.items()):
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error getting resources: {e}")
        
        print()
        
        # 4. 노드 상태
        print("4. Ray Nodes Status:")
        print("-" * 60)
        try:
            nodes = ray.nodes()
            print(f"Total nodes: {len(nodes)}")
            for node in nodes:
                node_id = node.get('NodeID', 'N/A')
                is_alive = node.get('Alive', False)
                resources = node.get('Resources', {})
                print(f"  Node {node_id}: Alive={is_alive}, Resources={resources}")
        except Exception as e:
            print(f"Error getting nodes: {e}")
        
        print()
        print("=" * 60)
        print("Tips:")
        print("1. Ray Dashboard에서 더 자세한 정보 확인: ray status로 URL 확인")
        print("2. 특정 actor의 로그: ray logs <actor_id>")
        print("3. 프로세스 스택 트레이스: kill -USR1 <pid>")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error connecting to Ray: {e}")
        print("\nMake sure Ray is running and you can connect to the cluster.")
        sys.exit(1)

if __name__ == "__main__":
    main()
