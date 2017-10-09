

class Path(object):

#定义Path，每一个Path类代表到达当前step的一个路径，如果是在init位置，path_history为空，则赋值当前step

    def __init__(self,step,goal,path_history = None):
        self.step = step
        #第一个点
        if not path_history:
            self.path_memory = [step]
        else:
            self.path_memory = path_history.path_memory + [step]

        self.heuristic = ( (goal[0] - step[0]) ** 2 + (goal[1] - step[1]) **2 ) **0.5
        self.gn_astar = len(self.path_memory)
        self.gn_greedy = 0
        self.fn_astar = self.heuristic + self.gn_astar
        self.fn_greedy = self.heuristic + self.gn_greedy


def path_explore(step):
    x_move = [-1,0,1,1,1,0,-1,-1]
    y_move = [1,1,1,0,-1,-1,-1,0]
    step_explored = []
    for i in range(8):
        newstep = [step[0]+x_move[i],step[1]+y_move[i]]
        step_explored.append(newstep)
    return step_explored

#边界判定
def boundary(step,mapsize,blocks):
    if (step[0]>0 and step[0]<mapsize[0] and step[1]>0 and step[1]<mapsize[1]\
        and (step not in blocks)):
        return True
    else:
        return False



def greedysearch(mapsize, blocks, init, goal):
    """Returns:
    - path: a list with length n (the number of steps) and each
    element of the list is a position in the format (xi, yi).
    Or a nx2 matrix.
    """
    init_path = Path(init,goal)
    frontier = [init_path]
    explored = []
    blocklist = list(list(i) for i in blocks)
    while True:
        frontier = sorted(frontier, key=lambda x: x.fn_greedy)
        current_path = frontier.pop(0)
        if current_path.heuristic == 0:
            return current_path.path_memory

        step_list = path_explore(current_path.step)
        for step in step_list:
            bound = boundary(step,mapsize,blocklist)
            if (bound and (step not in explored)):
                path_new = Path(step, goal, current_path)
                explored.append(step)
                frontier.append(path_new)




def astarsearch(mapsize, blocks, init, goal):
    """Returns:
    - path: a list with length n (the number of steps) and each
    element of the list is a position in the format (xi, yi).
    Or a nx2 matrix.
    """

    init_path = Path(init, goal)
    frontier = [init_path]
    explored = []
    blocklist = list(list(i) for i in blocks)
    while True:
        frontier = sorted(frontier, key=lambda x: x.fn_astar)
        current_path = frontier.pop(0)
        if current_path.heuristic == 0:
            return current_path.path_memory

        step_list = path_explore(current_path.step)
        for step in step_list:
            if boundary(step, mapsize, blocklist) and step not in explored:
                path_new = Path(step, goal, current_path)
                explored.append(step)
                frontier.append(path_new)
