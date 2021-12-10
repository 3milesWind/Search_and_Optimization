### Policy Iteration on Tree(n children) MDP
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

parser = argparse.ArgumentParser()
parser.add_argument("--n_children", type=int, default= 2, help="number of children of every node")
parser.add_argument("--n_depth", type=int, default= 4, help="depth of tree")
parser.add_argument("--gamma", type=float, default= 0.1, help="decay rate of reward")
args = parser.parse_args()
n_children = args.n_children
n_depth = args.n_depth
gamma = args.gamma

CHANCE = 0
STATE = 1

ims = []
frames = []
fig = plt.figure(figsize=(16, 7))

random.seed(1)

class Node():
    def __init__(self, type, depth, reward, prob, x, y):
        self.type = type  # MAX or CHANCE
        self.depth = depth
        self.children = []
        self.reward = reward
        self.prob = prob
        self.x = x
        self.y = y

        if self.type == STATE:
            self.value = 0
            self.plt_value_text = None
            self.policy = None
            self.plt_policy_arrow = None



class PolicyIter():
    def __init__(self, tree_depth):
        self.tree_depth = tree_depth
        self.root = Node(STATE, 0, random.randint(1, 10), None, 0, 0)

        self.build_tree(self.root)
        self.draw_tree(self.root)

    def build_tree(self, node):
        interval = 10 * n_children ** (self.tree_depth-node.depth)
        pos_list = [-(n_children-1)*interval/2]
        for _ in range(n_children-1):
            pos_list += [pos_list[-1]+interval]

        if node.depth == self.tree_depth:
            return

        if node.type == STATE:
            for i in range(n_children):
                node.children.append(Node(CHANCE, node.depth + 1, None, None, node.x + pos_list[i], node.y-10))
            ## Default use the first action as initial policy
            node.policy = node.children[0]

        else:
            ### Generate transition probabilities
            prob_list = [random.random() for _ in range(3)]
            s = sum(prob_list)
            prob_list = [i / s for i in prob_list]
            for i in range(n_children):
                node.children.append(Node(STATE, node.depth + 1, random.randint(1, 9), prob_list[i], node.x + pos_list[i], node.y - 10))

        for child in node.children:
            self.build_tree(child)


    def draw_tree(self, node):
        if node.type == STATE:
            ## plot node
            color = 'g'
            plt.scatter(node.x, node.y, c=color, marker='o', s=400)
            ## plot reward
            plt.text(node.x + 5, node.y, "%d" % node.reward, fontsize=10)
            for child in node.children:
                ## plot arrow
                plt.arrow(node.x, node.y - 1, child.x - node.x, child.y + 2 - node.y)
                self.draw_tree(child)
            ## plot initial policy
            if node.depth != self.tree_depth:
                node.plt_policy_arrow = plt.arrow(node.x, node.y - 1, node.policy.x - node.x,
                                                  node.policy.y + 2 - node.y, ec='r', width=0.1)
                frames.append(node.plt_policy_arrow)
                ims.append(frames.copy())


        elif node.type == CHANCE:
            ## plot node
            color = 'b'
            plt.scatter(node.x, node.y, c=color, marker='o', s=400)
            for child in node.children:
                ## plot arrow and policy arrow
                plt.arrow(node.x, node.y - 1, child.x - node.x, child.y + 2 - node.y)
                ## plot transition prob
                plt.text((node.x + child.x) / 2 -4, (node.y + child.y) / 2, "%.2f" % child.prob, fontsize=10)
                self.draw_tree(child)

        return


    def Bellman_update_with_policy(self, node, gamma):
        if node.type == STATE: ## Bellman update on state
            if node.depth != self.tree_depth:
                node.value = node.reward + gamma * sum([child.prob * child.value for child in node.policy.children])
            else:
                node.value = node.reward

            ## plot value
            plt1 = plt.scatter(node.x, node.y, c='r', marker='o', s=400)
            frames.append(plt1)

            if node.plt_value_text == None:
                node.plt_value_text = plt.text(node.x - 10.5, node.y, "%.2f" % node.value, c='r', fontsize=10)
            else:
                frames.remove(node.plt_value_text)
                node.plt_value_text = plt.text(node.x - 10.5, node.y, "%.2f" % node.value, c='r', fontsize=10)

            frames.append(node.plt_value_text)
            ims.append(frames.copy())

            plt3 = plt.scatter(node.x, node.y, c='g', marker='o', s=400)
            frames.append(plt3)
            ims.append(frames.copy())

        ## recursion
        for child in node.children:
            self.Bellman_update_with_policy(child, gamma)


    def update_policy(self, node):
        if node.type == STATE:
            best_action = None
            for action in node.children:
                current = sum([child.prob * child.value for child in node.policy.children])
                if sum([child.prob*child.value for child in action.children]) > current:
                    best_action = action
                    self.unchanged = False

            if best_action:
                node.policy = best_action

                frames.remove(node.plt_policy_arrow)
                node.plt_policy_arrow = plt.arrow(node.x, node.y - 1, best_action.x - node.x,
                                                  best_action.y + 2 - node.y, ec='r', width=0.1)

                frames.append(node.plt_policy_arrow)
                ims.append(frames.copy())

        for child in node.children:
            self.update_policy(child)



    def policy_iteration(self, gamma):
        round = 0
        while True:
            print('round',round)
            plt1 = plt.text(-(n_children**n_depth-1)*10 / 2, 0, "Round: %d"%round, fontsize=20, bbox={'fc':'w', 'ec':'k'})
            frames.append(plt1)
            ims.append(frames.copy())

            self.unchanged = True
            self.Bellman_update_with_policy(self.root, gamma)
            self.update_policy(self.root)

            if self.unchanged:
                break
            round += 1

        print(self.root.value)
        return


def main():
    plt.axis('off')
    plt.tight_layout()

    VI = PolicyIter(n_depth)
    VI.policy_iteration(gamma)

    ani = animation.ArtistAnimation(fig, ims, interval=1)
    writer = PillowWriter(fps=3)
    ani.save("./gif/policy_iter.gif", writer=writer)
    plt.show()

if __name__ == "__main__":
    main()