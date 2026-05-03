
import json

class Node:
    def __init__(self, node_id, title, text):
        self.node_id = node_id
        self.title = title
        self.text = text
        self.children = []
        self.parent = None

def build_tree(path):
    with open(path, "r") as f:
        data = json.load(f)

    nodes = {}
    for item in data:
        nodes[item['node_id']] = Node(
            item['node_id'],
            item['title'],
            item['text_content']
        )

    root = None
    for item in data:
        node = nodes[item['node_id']]
        if item['parent_id'] is None:
            root = node
        else:
            parent = nodes[item['parent_id']]
            node.parent = parent
            parent.children.append(node)

    return root, list(nodes.values())
