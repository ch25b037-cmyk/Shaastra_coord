
from collections import deque

def bfs(node, depth=1):
    result = []
    queue = deque([(node, 0)])
    while queue:
        current, d = queue.popleft()
        result.append(current)
        if d < depth:
            for child in current.children:
                queue.append((child, d+1))
    return result

def dfs(node, depth=1):
    result = []
    def _dfs(n, d):
        result.append(n)
        if d < depth:
            for c in n.children:
                _dfs(c, d+1)
    _dfs(node, 0)
    return result
