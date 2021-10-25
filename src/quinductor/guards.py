import os
import re
from collections import defaultdict

import numpy as np

from .rules import *


def generate_guards(templates, temp_fname, guard_fname="gen_guards.txt"):
    guards = defaultdict(list)
    temp_fname = os.path.splitext(os.path.basename(temp_fname))[0]
    negate = False
    for i, temp in enumerate(templates):
        guard_set = []
        pos_tag, morph, has_aux = [], [], []

        for p in temp['props']:
            pos_tag.append(p['pos'])
            has_aux.append(p['has_aux'])
            morph.append(str(p['feats']).strip())

        pos_tag, morph = set(pos_tag), set(morph)

        has_aux = all(has_aux)
        for pos, mrph in zip(pos_tag, morph):
            guard = []
            if pos:
                guard.append(f"n.pos is {pos}")
            if mrph:
                guard.append(f"n.morph has {mrph}")
            guard_set.append(guard)

        guard = []
        arity, aux_not_exists = defaultdict(set), None
        for j, clause in enumerate([temp['question'], temp['answer']]):
            for el in clause.strip().split():
                if el.startswith('[w.lemma'):
                    continue

                if el.endswith('.lemma]'):
                    el = el.replace('.lemma', '')


                # TODO: remove the dependency on the template word being called `w` (make it an argument)
                if el == '[w]' and pos_tag == 'VERB' and not has_aux and aux_not_exists != 'NA':
                    aux_not_exists = 'INS'

                if el.startswith('[w.aux') or el.startswith('<w.aux'):
                    aux_not_exists = 'NA'

                # ignore negations, cuz they're not requirements
                if el.endswith(">") and negate:
                    negate = False
                    continue
                if negate: continue
                if el.strip() == "-":
                    negate = True
                    continue

                parts = el.split("-")[0].strip().split(".")
                if len(parts) >= 2:
                    gg = "n"
                    for part in parts[1:]:
                        p = re.sub(r'#[0-9]+', '', part.strip())
                        gg += ".{}".format(p.replace('>', '').replace(']', ''))
                    arity[gg].add(".".join(parts[1:]).replace('>', '').replace(']', ''))
                    if (parts[1].strip().startswith('nsubj') and len(parts) == 2) or (j == 1 and parts[-1].strip().startswith('nsubj')):
                        # either the clause has nsubj as a direct descendant of root
                        # or the answer points to subject of any clause in the sentence
                        guard.insert(0, f'{gg}.morph has_not PronType=Rel')

        for g, a in arity.items():
            N = len(a)
            if N > 1:
                guard.append(f"{g}/{N} exists")
            else:
                guard.append(f"{g} exists")

        if aux_not_exists == 'INS':
            guard.append('n.aux not_exists')

        for g in guard_set:
            new_guard = ", ".join(g + guard)
            guards[new_guard.strip()].append("{}{}".format(temp_fname, i+1))

    with open(guard_fname, "w") as f:
        for g, t in guards.items():
            f.write("{} -> {}\n".format(g, ",".join(t)))
    return guards



class GuardTreeNode:
    def __init__(self, rule, cond):
        self.__cond = cond
        self.__rule = rule
        self.__children = dict()
        self.templates = []
        self.parent = None

    def add_child(self, c):
        if c.rule in self.children:
            obj = self.children[c.rule]
            # this is just a safety code, since this should not happen
            for ch in obj.children.values():
                c.add_child(ch)
            c.templates.extend(obj.templates)
        c.parent = self
        self.children[c.rule] = c

    @property
    def children(self):
        return self.__children

    def is_root(self):
        return self.__rule == "ROOT" and self.__cond is None

    @property
    def rule(self):
        return self.__rule

    def cond(self, n, check_subtrees=False):
        return self.__cond(n, check_subtrees)

    @classmethod
    def init(cls, gtn):
        return cls(gtn.rule, gtn.cond)

    def print_tree(self, indent=0):
        print(" " * indent + str(self))
        for c in self.__children.values():
            c.print_tree(indent + 2)


    def __str__(self):
        return f"{self.__rule}___{(','.join(self.templates) or 'no_templates')}___{len(self.__children)}"


def load_guards(files):
    def pair(g_from, g_to, templates=None):
        g_to = GuardTreeNode.init(g_to)
        if templates:
            g_to.templates.extend(templates)
        g_from.add_child(g_to)
        g_from = g_to
        return g_from

    # we assume that we get udon2.Node as an input for every guard
    guard_nodes, guard_freq, parsed_guards = {}, defaultdict(int), []
    for fname in files:
        with open(fname) as f:
            for rule in f:
                guard = parse(rule, Guard)
                parsed_guards.append(guard)

                for ub in guard.uclauses:
                    guard_rule, guard_cond = str(ub), None
                    if guard_rule not in guard_nodes:
                        if ub.cond == 'exists':
                            if hasattr(ub, 'arity'):
                                try:
                                    arity = int(ub.arity)
                                except:
                                    continue
                                # Need to fix this for checking subtrees
                                guard_cond = lambda n,cs,v=ub.tag,a=arity: check_arity(n, v, a, check_subtrees=cs)
                            else:
                                guard_cond = lambda n,cs,v=ub.tag: relation_exists(n, v, check_subtrees=cs)
                        elif ub.cond == 'not_exists':
                            guard_cond = lambda n,cs,v=ub.tag: relation_not_exists(n, v, check_subtrees=cs)

                        if guard_cond:
                            gtn = GuardTreeNode(guard_rule, guard_cond)
                            guard_nodes[guard_rule] = gtn
                    guard_freq[guard_rule] += 1

                for gb in guard.bclauses:
                    guard_rule, guard_cond = str(gb), None

                    tag_chain_cond = lambda x,cs: x
                    if gb.tag:
                        tag_chain_cond = lambda x,cs,v=gb.tag: filter_by_rel_chain(x, v, check_subtrees=cs)

                    if guard_rule not in guard_nodes:
                        if gb.prop == 'pos':
                            if gb.cond == 'is':
                                guard_cond = lambda n,cs,v=gb.val,f=tag_chain_cond: filter_by_pos(n, v, f=f, check_subtrees=cs)
                            elif gb.cond == 'is_not':
                                guard_cond = lambda n,cs,v=gb.val,f=tag_chain_cond: filter_by_pos(n, v, f=f, check_subtrees=cs, negate=True)
                            elif gb.cond == 'is_in':
                                pass
                        # elif gb.prop == 'rel':
                        #     if gb.cond == 'is':
                        #         guard_cond = lambda n,v=gb.val,f=tag_chain_cond: filter_by_rel(f(n), v)
                        #     elif gb.cond == 'is_not':
                        #         guard_cond = lambda n,v=gb.val,f=tag_chain_cond: filter_by_rel(f(n), v, negate=True)
                        #     elif gb.cond == 'is_in':
                        #         pass
                        elif gb.prop == 'morph':
                            if gb.cond == 'has':
                                guard_cond = lambda n,cs,v=gb.val,f=tag_chain_cond: filter_by_morph(n, v, f=f, check_subtrees=cs)
                            elif gb.cond == 'has_not':
                                guard_cond = lambda n,cs,v=gb.val,f=tag_chain_cond: filter_by_morph(n, v, f=f, check_subtrees=cs, negate=True)
                        if guard_cond:
                            gtn = GuardTreeNode(guard_rule, guard_cond)
                            guard_nodes[guard_rule] = gtn
                    guard_freq[guard_rule] += 1

    # print(guard_freq)
    direct_children = set()
    for j, p in enumerate(parsed_guards):
        clauses = list(set(map(str, p.uclauses))) + list(set(map(str, p.bclauses)))

        if len(clauses) == 1:
            g = GuardTreeNode.init(guard_nodes[clauses[0]])
            g.templates.extend(p.templates.split(','))
            direct_children.add(g)
        else:
            freqs = [guard_freq[x] for x in clauses]
            ind = np.argsort(freqs)[::-1] # reverse argsort

            direct_children.add(guard_nodes[clauses[ind[0]]])
            if len(clauses) > 2:
                g_from = pair(guard_nodes[clauses[ind[0]]], guard_nodes[clauses[ind[1]]])
                for i in ind[2:-1]:
                    g_from = pair(g_from, guard_nodes[clauses[i]])
                g_from = pair(g_from, guard_nodes[clauses[ind[-1]]], p.templates.split(','))
            else:
                g_from = pair(guard_nodes[clauses[ind[0]]], guard_nodes[clauses[ind[1]]], p.templates.split(','))

    root = GuardTreeNode("ROOT", None)
    for gn in direct_children:
        root.add_child(gn)
    return root