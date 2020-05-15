import argparse
import gzip
import json
import random
import ast
#import operator as op

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("-c", "--components", dest="components", type=int, default=1000, help="Number of graph components to generate")
    args = parser.parse_args()

    #operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    #             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
    #             ast.USub: op.neg}
    binops = [ast.Add, ast.Sub, ast.Mult, ast.Div] #, ast.Pow]
    unops = [ast.UAdd, ast.USub]
    

    #mode="eval"
    # entities = []
    # geom = 0.1
    # max_size = 20
    # pA = 0.5
    offset = 0
    entities = []
    for i in range(args.components):
        num_consts = random.randint(3, 10)
        consts = [random.random() * 20.0 - 10.0 for _ in range(num_consts)]
        #entities += [{"value" : c, "entity_type" : "node", "id" : i + offset} for i, c in enumerate(consts)]
        #nodes = [ast.Num(c, lineno=0, col_offset=0) for c in consts]
        nodes = []
        for i, c in enumerate(consts):
            node = ast.Num(c, lineno=0, col_offset=0)
            entity = {"value" : c, "entity_type" : "node", "name" : "constant", "id" : len(entities)}
            entities.append(entity)
            node.id = entity["id"]
            nodes.append(node)
        while len(nodes) > 1:
            unary = random.random() < 0.1
            if unary:
                op = random.choice(unops)
                nodei = random.randint(0, len(nodes) - 1)
                node = nodes[nodei]
                new_node = ast.UnaryOp(op(), node, lineno=0, col_offset=0)
                entity = {"entity_type" : "node", "name" : str(type(op)), "id" : len(entities)}
                new_node.id = len(entities)
                entities.append(entity)
                entities[node.id]["unary_arg"] = str(new_node.id) #entities[-1]["id"]
            else:
                op = random.choice(binops)
                nodeAi = random.randint(0, len(nodes) - 1)
                nodeBi = random.randint(0, len(nodes) - 2)
                nodeBi = (nodeBi + 1) if nodeAi == nodeBi else nodeBi                
                nodeA = nodes[nodeAi]
                nodeB = nodes[nodeBi]
                new_node = ast.BinOp(nodeA, op(), nodeB, lineno=0, col_offset=0)
                entity = {"entity_type" : "node", "name" : str(type(op)), "id" : len(entities)}
                new_node.id = len(entities)
                entities.append(entity)
                entities[nodeA.id]["left_arg"] = str(new_node.id) #entities[-1]["id"]
                entities[nodeB.id]["right_arg"] = str(new_node.id) #entities[-1]["id"]
                nodes = [n for i, n in enumerate(nodes) if i not in [nodeAi, nodeBi]] + [new_node]
            pass
        e = ast.Expression(nodes[0], lineno=0, col_offset=0)
        v = eval(compile(e, filename="<AST>", mode="eval"))
        entities[-1]["value"] = v
        #components.append(entities)
        #print(v)
        #print(entities)
    #     component = {"a" : [], "b" : []}
    #     for _ in range(max_size):
    #         a_cat = random.randint(0, 10)
    #         b_cat = random.randint(0, 10)
    #         entity = {"entity_type" : "a" if random.random() < pA else "b", "id" : len(entities)}
    #         entity["{}_numeric".format(entity["entity_type"])] = random.randint(0, 10)
    #         component[entity["entity_type"]].append(entity["id"])
    #         entities.append(entity)
    #         if random.random() < geom:
    #             break
    #         #ra = random.randint(0, 20)
    #         #rb = random.randint(0, 20)

    #         #fields["a_label"] = "a_{}".format(a_cat)
    #         #a_c1, a_c2 = ("4", "3") if b_cat > 5 else ("2", "1")
    #         #fields["a_categorical"] = "b_{}".format(b_cat)
    #         #fields["a_numeric"] = b_cat
    #         #fields["a_sequential"] = "".join([a_c1] * ra + [a_c2] * rb)

    #         #fields["b_label"] = "b_{}".format(b_cat)
    #         #b_c1, b_c2 = ("1", "2") if a_cat > 5 else ("3", "4")
    #         #fields["b_categorical"] = "a_{}".format(a_cat)
    #         #fields["b_numeric"] = a_cat
    #         #fields["b_sequential"] = "".join([b_c1] * ra + [b_c2] * rb)

    with gzip.open(args.output, "wt") as ofd:
        for entity in entities:
            entity["id"] = str(entity["id"])
            ofd.write(json.dumps(entity) + "\n")
