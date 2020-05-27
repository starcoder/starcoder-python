import argparse
import gzip
import json
import random
import ast


#operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
#             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
#             ast.USub: op.neg}
binops = [(ast.Add, "add", lambda x, y : x + y),
          (ast.Sub, "subtract", lambda x, y : x - y),
          (ast.Mult, "multiply", lambda x, y : x * y),
          (ast.Div, "divide", lambda x, y : x / y),
]
binops_dict = {k : (n, v) for k, n, v in binops}


unops = [
    (ast.UAdd, "id", lambda x : x),
    (ast.USub, "negate", lambda x : -x),
]
unops_dict = {k : (n, v) for k, n, v in unops}


def random_tree(min_count, max_count, unary_prob, prefix):
    consts = generate_constants(min_count, max_count)
    tree = generate_tree(consts, unary_prob=unary_prob)
    ptree = populate_tree(tree, prefix)
    jtree = to_json(ptree)
    return jtree


def generate_constants(minimum_constants, maximum_constants):
    num_consts = random.randint(minimum_constants, maximum_constants)
    consts = [random.random() for _ in range(num_consts)]
    nodes = []
    for node_id, c in enumerate(consts):
        node = ast.Num(c, lineno=0, col_offset=0)
        nodes.append(node)
    return nodes
    

def generate_tree(nodes, unary_prob=0.0):
    if len(nodes) == 1:
        return ast.Expression(nodes[0], lineno=0, col_offset=0)
    else:
        unary = random.random() < unary_prob
        if unary:
            op, name, _ = random.choice(unops)
            arg_node_idx = random.randint(0, len(nodes) - 1)
            arg_node = nodes[arg_node_idx]
            new_node = ast.UnaryOp(op(), arg_node, lineno=0, col_offset=0)
            remove = [arg_node_idx]
        else:
            op, name, _ = random.choice(binops)
            left_node_idx = random.randint(0, len(nodes) - 1)
            right_node_idx = random.randint(0, len(nodes) - 2)
            right_node_idx = (right_node_idx + 1) if left_node_idx == right_node_idx else right_node_idx
            left_node = nodes[left_node_idx]
            right_node = nodes[right_node_idx]
            new_node = ast.BinOp(left_node, op(), right_node, lineno=0, col_offset=0)
            remove = [left_node_idx, right_node_idx]
        return generate_tree([n for i, n in enumerate(nodes) if i not in remove] + [new_node], unary_prob)


def render_tree(node):
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            op = "-"
        elif isinstance(node.op, ast.UAdd):
            op = "+"
        return "{}{}".format(op, render_tree(node.operand))
    elif isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Sub):
            op = "-"
        elif isinstance(node.op, ast.Add):
            op = "+"
        elif isinstance(node.op, ast.Mult):
            op = "*"
        elif isinstance(node.op, ast.Div):
            op = "/"
        return "({} {} {})".format(render_tree(node.left), op, render_tree(node.right))
    elif isinstance(node, ast.Num):
        return "{}".format(node.n)

    
def populate_tree(node, prefix=""):
    if isinstance(node, ast.Expression):
        populate_tree(node.body, "{}_1".format(prefix))
        return node
    elif isinstance(node, ast.Expr):
        populate_tree(node.value, "{}_1".format(prefix))
        return node    
    elif isinstance(node, ast.BinOp):
        name, op = binops_dict[type(node.op)]
        node.value = op(populate_tree(node.left, "{}_1".format(prefix)), populate_tree(node.right, "{}_2".format(prefix)))
        node.id = prefix
        node.name = name
        node.left.relations = {"left_arg_for" : node.id}
        node.right.relations = {"right_arg_for" : node.id}
    elif isinstance(node, ast.UnaryOp):
        name, op = unops_dict[type(node.op)]
        node.value = op(populate_tree(node.operand, "{}_1".format(prefix)))
        node.id = prefix
        node.name = name
        node.operand.relations = {"unary_arg_for" : node.id}
    elif isinstance(node, ast.Num):
        node.value = node.n
        node.id = prefix
        node.name = "const"
    return node.value


def flatten_tree(node, component_id):
    if hasattr(node, "id"):        
        node.id = "{}_{}".format(component_id, node.id)
    node.relations = {}
    if isinstance(node, ast.Num):        
        return [node]
    elif isinstance(node, ast.UnaryOp):
        node.relations["unary_arg"] = "{}_{}".format(component_id, node.operand.id)
        return flatten_tree(node.operand, component_id) + [node]
    elif isinstance(node, ast.BinOp):
        node.relations["left_arg"] = "{}_{}".format(component_id, node.left.id)
        node.relations["right_arg"] = "{}_{}".format(component_id, node.right.id)
        return flatten_tree(node.left, component_id) + flatten_tree(node.right, component_id) + [node]
    else:
        if hasattr(node, "body"):
            return flatten_tree(node.body, component_id)
        else:
            return flatten_tree(node.value, component_id)

        
def grow_tree(nodes, count, unary_prob=0.1):
    if len(nodes) == 1:
        return ast.Expression(nodes[0], lineno=0, col_offset=0)
    else:
        unary = random.random() < unary_prob
        if unary and False:
            op, name, lam = random.choice(unops)
            arg_node_idx = random.randint(0, len(nodes) - 1)
            arg_node = nodes[arg_node_idx]
            new_node = ast.UnaryOp(op(), arg_node, lineno=0, col_offset=0)
            new_node.value = lam(arg_node.value)
            remove = [arg_node_idx]
        else:
            op, name, lam = random.choice(binops)
            left_node_idx = random.randint(0, len(nodes) - 1)
            right_node_idx = random.randint(0, len(nodes) - 2)
            right_node_idx = (right_node_idx + 1) if left_node_idx == right_node_idx else right_node_idx
            left_node = nodes[left_node_idx]
            right_node = nodes[right_node_idx]
            new_node = ast.BinOp(left_node, op(), right_node, lineno=0, col_offset=0)
            new_node.value = lam(left_node.value, right_node.value)
            remove = [left_node_idx, right_node_idx]
        new_node.id = count
        return grow_tree([n for i, n in enumerate(nodes) if i not in remove] + [new_node], count + 1, unary_prob)


def reverse_edges(component, mappings):
    node_lookup = {n.id : n for n in component}
    for node in component:
        for k, v in mappings.items():
            if k in node.relations:
                other_id = node.relations[k]
                del node_lookup[node.id].relations[k]
                node_lookup[other_id].relations[v] = node.id                
    return node_lookup.values()


def to_json(tree):
    retval = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Num, ast.BinOp, ast.UnaryOp)):
            retval.append(dict([
                ("id", node.id),
                ("name", node.name),                
                ("value", node.value),
                ("entity_type", "node"),
            ] + ([(k, v) for k, v in node.relations.items()] if hasattr(node, "relations") else [])
            ))
    return retval
