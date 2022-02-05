import util

class DFS(object):
    def depthFirstSearch(self, problem):
     
        cnt_node = util.Stack()
        cmptd_node = []
        
        startState = problem.getStartState()
        begin_node = (startState, [])
        
        cnt_node.push(begin_node)
        
        while not cnt_node.isEmpty():
            
            cnt_ste, actions = cnt_node.pop()
            
            if cnt_ste not in cmptd_node:
                
                cmptd_node.append(cnt_ste)

                if problem.isGoalState(cnt_ste):
                    return actions
                else:
                    
                    
                    successors = problem.getSuccessors(cnt_ste)
                    
                    
                    for sucr_ste, sucr_act, sucr_cst in successors:
                        n_act = actions + [sucr_act]
                        n_node = (sucr_ste, n_act)
                        cnt_node.push(n_node)

        return actions

class BFS(object):
    def breadthFirstSearch(self, problem):
        
    
        cnt_node = util.Queue()
        cmptd_node = []
    
        startState = problem.getStartState()
        begin_node = (startState, []) 
    
        cnt_node.push(begin_node)
    
        while not cnt_node.isEmpty():
        
            cnt_ste, actions= cnt_node.pop()
       
            if (cnt_ste not in cmptd_node):
            
              cmptd_node.append(cnt_ste)

              if problem.isGoalState(cnt_ste):
                  return actions
              else:
                
                  successors = problem.getSuccessors(cnt_ste)
                
                  for sucr_ste, sucr_act, sucr_cst in successors:
                    n_act = actions + [sucr_act]
                   
                    n_node = (sucr_ste, n_act)

                    cnt_node.push(n_node)

        return actions 

class UCS(object):
    def uniformCostSearch(self, problem):
        
    
        cnt_node = util.PriorityQueue()
        cmptd_node = {}
    
        startState = problem.getStartState()
        begin_node = (startState, [], 0) 
    
        cnt_node.push(begin_node, 0)
    
        while not cnt_node.isEmpty():
        
            cnt_ste, actions, Cost = cnt_node.pop()
       
            if (cnt_ste not in cmptd_node):
            
               cmptd_node[cnt_ste] = Cost

               if problem.isGoalState(cnt_ste):
                  return actions
               else:
                
                  successors = problem.getSuccessors(cnt_ste)
                
                  for sucr_ste, sucr_act, sucr_cst in successors:
                    n_act = actions + [sucr_act]
                    newCost = Cost + sucr_cst
                    n_node = (sucr_ste, n_act, newCost)

                    cnt_node.update(n_node, newCost)

        return actions
        util.raiseNotDefined()
        
class aSearch (object):
    def nullHeuristic( state, problem=None):
        
        return 0
    def aStarSearch(self,problem, heuristic=nullHeuristic):
       
        cnt_node = util.PriorityQueue()
        cmptd_node = [] 

        startState = problem.getStartState()
        begin_node = (startState, [], 0) 
        cnt_node.push(begin_node, 0)

        while not cnt_node.isEmpty():

        
           cnt_ste, actions, Cost = cnt_node.pop()

        
           currentNode = (cnt_ste, Cost)
           cmptd_node.append((cnt_ste, Cost))

           if problem.isGoalState(cnt_ste):
              return actions

           else:
            
              successors = problem.getSuccessors(cnt_ste)

            
              for sucr_ste, sucr_act, sucr_cst in successors:
                n_act = actions + [sucr_act]
                newCost = problem.getCostOfActions(n_act)
                n_node = (sucr_ste, n_act, newCost)

                
                ardy_vst = False
                for visited in cmptd_node:
                    
                    vstd_ste, vst_cst = visited

                    if (sucr_ste == vstd_ste) and (newCost >= vst_cst):
                        ardy_vst = True

                
                if not ardy_vst:
                    cnt_node.push(n_node, newCost + heuristic(sucr_ste, problem))
                    cmptd_node.append((sucr_ste, newCost))

        return actions
        util.raiseNotDefined()

