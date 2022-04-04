# -*- coding: utf-8 -*-
import os
import re
import json
from collections import defaultdict
from itertools import product, combinations
from pypeg2 import *

from .common import chain_to_root


telugu_chars = '\u0C00-\u0C7F'
arabic_chars = '\u0600-\u06FF\ufb50-\ufdff\ufe70-\ufefc'
japanese_chars = '\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf'
question_word = re.compile(r'[{0}{1}{2}\w\?,\.]+(\'/[{0}{1}{2}\w\?,\.]+)*'.format(japanese_chars, arabic_chars, telugu_chars), re.U)
inter_word = re.compile(r'[{0}{1}{2}\w0-9\'\-\?,\.]+'.format(japanese_chars, arabic_chars, telugu_chars), re.U)


# to be fixed later with UDon2
base_folder = os.path.dirname(__file__)
lang_spec = json.load(open(os.path.join(base_folder, 'lang_spec_feats.json')))
lang_spec_tags = [x for d in lang_spec.values() for x in d.keys()]
lang_spec_vals = [y for d in lang_spec.values() for x in d.values() for y in x]


class PosTag(Keyword):
    """
    A keyword (enum), describing a part-of-speech tag, should be capitalized
    """
    VALUES = ["ADJ", "ADP", "PUNCT", "ADV", "AUX", "SYM", "INTJ", "CCONJ", "X", "NOUN", "DET", "PROPN", "NUM", "VERB", "PART", "PRON", "SCONJ"]
    grammar = Enum(*(K(v) for v in VALUES))


class MorphTag(Keyword):
    VALUES = ["PronType", "Gender", "VerbForm", "NumType", "NumForm", "Animacy", "Mood", "Poss", "NounClass", "Tense", "Reflex", "Number", "Aspect",
              "Case", "Voice", "Abbr", "Definite", "Evident", "Typo", "Degree", "Polarity", "Person", "Polite", "Clusivity", "Variant",
              "Derivation", "Foreign", "InfForm", "PartForm", "Person[psor]", "Number[psor]"] + lang_spec_tags
    grammar = Enum(*(K(v) for v in VALUES))
    regex = re.compile(r'[\w\[\]]+', re.U)


class MorphValue(Keyword):
    VALUES = ['Ex', 'Bantu16', 'Ben', 'Pass', 'Dis', 'Hab', 'Cau', 'Ger', 'Qot', 'Abs', 'Neg', 'Cons', 'Bantu18', 'Humb', 'Gen', 'Cmp', 'Mult',
              'Sing', 'Cns', 'Tra', 'Part', 'Nom', 'Hum', 'All', 'Pauc', 'Fem', 'Masc', 'Bantu8', 'Sup', 'Ess', 'Coll', 'Acc', 'Conv', 'Abe',
              'Prp', 'Dat', 'Grpa', 'Count', 'Equ', 'Infm', 'Jus', 'Inf', 'Adm', '1', 'Abl', 'Gdv', 'Com', 'Sets', 'Rel', 'Prosp', 'Dir',
              'Bantu13', '4', 'Fh', 'Pres', 'Bantu20', 'Int', 'Rcp', 'Per', 'Bantu2', 'Ade', 'Tot', 'Ptan', 'Form', 'Plur', 'Ter', 'Loc', 'Bantu1',
              'Frac', 'Anim', 'Bantu17', '0', 'Dual', 'Pot', 'Iter', 'Bantu15', 'Bantu11', 'Opt', 'Range', 'Fin', '3', 'Bantu3', 'Elev', 'Exc',
              '2', 'Dist', 'Bantu5', 'Prog', 'Pqp', 'Vnoun', 'Bantu4', 'Nfh', 'Emp', 'In', 'Pos', 'Des', 'Inan', 'Ord', 'Del', 'Add', 'Def', 'Ine',
              'Bantu10', 'Act', 'Bantu19', 'Ill', 'Imp', 'Art', 'Inv', 'Voc', 'Neut', 'Ind', 'Bantu14', 'Mid', 'Dem', 'Par', 'Past', 'Card', 'Fut',
              'Prs', 'Grpl', 'Lat', 'Antip', 'Perf', 'Yes', 'Bantu7', 'Tem', 'Ela', 'Spec', 'Nec', 'Tri', 'Ins', 'Cnd', 'Erg', 'Bantu9', 'Bantu12',
              'Nhum', 'Bantu6', 'Sub', "Short", "Inen", "Ja", "Lainen", "Llinen", "Vs", "U", "Tar", "Digit", "Combi", "Roman", "Word"] + lang_spec_vals
    grammar = Enum(*(K(v) for v in VALUES))


class MorphValueChain(List):
    grammar = MorphValue, maybe_some(",", MorphValue)

    def __str__(self):
        return "|".join(map(str, list(self)))


class MorphKeyVal:
    grammar = attr('tag', MorphTag), '=', attr('val', MorphValueChain)

    def __str__(self):
        return f"{self.tag}={self.val}"


class MorphKeyValChain(List):
    grammar = MorphKeyVal, maybe_some("|", MorphKeyVal)

    def __str__(self):
        return "|".join(map(str, list(self)))


class Prop(Keyword):
    VALUES = ['pos', 'rel', 'morph', 'text', 'lemma']
    grammar = Enum(*(K(v) for v in VALUES))


class DepRelTag(Keyword):
    CORE_NOMINALS = ["nsubj", "obj", "iobj"]
    CORE_CLAUSES = ["csubj", "ccomp", "xcomp"]
    NON_CORE_NOMINALS = ["obl", "vocative", "expl", "dislocated"]
    NON_CORE_CLAUSES = ["advcl"]
    NON_CORE_MODIFIERS = ["advmod", "discourse"]
    NON_CORE_FUNCTION = ["aux", "cop", "mark"]
    NOMINAL_DEP_NOMINALS = ["nmod", "appos", "nummod"]
    NOMINAL_DEP_CLAUSES = ["acl"]
    NOMINAL_DEP_MODIFIERS = ["amod"]
    NOMINAL_DEP_FUNCTION = ["det", "clf", "case"]

    COORDINATION = ["conj", "cc"]
    MWE = ["fixed", "flat", "compound"]
    LOOSE = ["list", "parataxis"]
    SPECIAL = ["orphan", "goeswith", "reparandum"]
    OTHER = ["punct", "root", "dep"]

    VALUES = CORE_NOMINALS + CORE_CLAUSES + NON_CORE_NOMINALS + NON_CORE_CLAUSES + \
            NON_CORE_MODIFIERS + NON_CORE_FUNCTION + NOMINAL_DEP_FUNCTION + NOMINAL_DEP_MODIFIERS + \
            NOMINAL_DEP_CLAUSES + NOMINAL_DEP_NOMINALS + COORDINATION + MWE + LOOSE + SPECIAL + OTHER
    grammar = Enum(*(K(v) for v in VALUES))


class DepRelModifier(Keyword):
    grammar = Enum(K("pass"), K("relcl"), K("agent"), K("prt"), K("name"), K("poss"), K("npmod"), K("ds"), K("gobj"), K("emph"),
                   K("tmod"), K("preconj"), K("foreign"), K("predet"), K("cleft"), K("gov"), K("cop"), K("nn"), K("own"), K("arg"),
                   K("svc"), K("nc"), K("gsubj"))


class ModifiedDepRelTag:
    grammar = attr('rel', DepRelTag), optional(':', attr('mod', DepRelModifier)), optional('#', attr('id', str)), optional(flag("exact", '*'))

    def __str__(self):
        s = self.rel
        if hasattr(self, 'mod'):
            s += ":" + self.mod
        if hasattr(self, 'exact') and self.exact:
            s += '*'
        return s


class DepTagChain(List):
    grammar = maybe_some(".", ModifiedDepRelTag)

    def __str__(self):
        return ".".join([str(a) for a in self])


class NegativeDepEntity:
    grammar = attr('tag', ModifiedDepRelTag), attr('chain', optional(DepTagChain))

    def __str__(self):
        return str(self.tag) + (("." + str(self.chain)) if hasattr(self, 'chain') and len(self.chain) > 0 else "")


class NegativeDepRel(List):
    grammar = maybe_some(optional(blank), '-', optional(blank), NegativeDepEntity)

    def __str__(self):
        return " - ".join([str(r) for r in self])


class UnaryCondition(Keyword):
    grammar = Enum(K("exists_once"), K("exists"), K("not_exists"))


class UnaryClause(Namespace):
    grammar = name(), attr('tag', DepTagChain), optional('/', attr('arity', str)), optional(blank), attr("cond", UnaryCondition)

    def __str__(self):
        return self.name + "." + ".".join([str(t) for t in list(self.tag)]) +\
            (f"/{self.arity}" if hasattr(self, 'arity') else "") +  " " + str(self.cond)


class BinaryCondition(Keyword):
    grammar = Enum(K("is"), K("is_not"), K("is_in"), K("has"), K("has_not"))


class BinaryClause(Namespace):
    grammar = name(), attr('tag', DepTagChain), optional('.', attr('prop', Prop)), optional(blank), attr('cond', BinaryCondition),\
        optional(blank), attr('val', [PosTag, DepRelTag, MorphKeyValChain])

    def __str__(self):
        return self.name + ("." + ".".join([str(t) for t in list(self.tag)]) if self.tag else "") + \
                "." + self.prop + " " + self.cond + " " + str(self.val)

class Guard(Namespace):
    grammar = attr('bclauses', maybe_some(csl(BinaryClause))), optional(optional(","), attr('uclauses', maybe_some(csl(UnaryClause)))),\
        optional(blank), "->", optional(blank), attr('templates', restline)

    def __str__(self):
        if self.bclauses and self.uclauses:
            return "{}, {} -> {}".format(", ".join(map(str, self.bclauses)), ", ".join(map(str, self.uclauses)), self.templates)
        elif self.bclauses:
            return "{} -> {}".format(", ".join(map(str, self.bclauses)), self.templates)
        else:
            return "{} -> {}".format(", ".join(map(str, self.uclauses)), self.templates)


class SubtreePlaceholder:
    grammar = '<', name(), optional(attr('tag', DepTagChain)), optional(attr('neg', NegativeDepRel)), '>'

    def __str__(self):
        return "<{}.{}{}>".format(
            self.name,
            ".".join([str(t) for t in list(self.tag)]),
            (" - {}".format(str(self.neg)) if hasattr(self, 'neg') and len(self.neg) > 0 else "")
        )


class NodePlaceholder:
    grammar = '[', name(), optional(attr('tag', DepTagChain)), optional(".", attr('prop', Prop)), ']'

    def __str__(self):
        return "[{}.{}]".format(self.name, ".".join([str(t) for t in list(self.tag)])) if self.tag else "[{}]".format(self.name)


class Template(List):
    grammar = some([question_word, inter_word, NodePlaceholder, SubtreePlaceholder]), optional(blank), "=>",\
        optional(blank), attr('answer', optional(some([inter_word, NodePlaceholder, SubtreePlaceholder])))

    def __str__(self):
        # NOTE: this will give you a non-normalized template (without IDs and possibly with repeating negatives)
        return "{} => {}".format(" ".join(map(str, self)), " ".join(map(str, self.answer)))
            
    def to_string(self, only_answer=False, only_question=False):
        if only_question:
            return " ".join(map(str, self))
        elif only_answer:
            return " ".join(map(str, self.answer))
        else:
            return str(self)


def filter_by_pos(node, value, f=None, check_subtrees=False, negate=False):
    def __filter_by_pos(n):
        result = []
        pos_match = n.upos == value
        if (pos_match and (not negate)) or ((not pos_match) and negate):
            result.append(n)


        if f is not None and identify_fn:
            # if we don't have any tags before we might possibly want to check the subtrees
            # otherwise they have been checked by `filter_by_rel_chain`, so all that's left 
            # is to check POS-tags on the nodes themselves, which we have already done above.
            if check_subtrees:
                # TOFIX: selectByPos returns udon2.NodeList, but want just a python list
                result.extend(n.select_by("upos", value, negate)) # search in the subtree except the POS
        return result

    identify_fn = True
    if f is not None:
        identity_fn = f(node, check_subtrees) == node
        node = f(node, check_subtrees)
    if type(node) == list:
        # see comment in filter_by_morph
        res = []
        for beg, end in node:
            filtered = __filter_by_pos(end)
            if filtered:
                res.append(beg)
    else:
        res = __filter_by_pos(node)
    return res
        

# TODO: when filtering by rel, recall that if node's rel satsifies the condition, then we want to have its parent as `n`
# def filter_by_rel(node, value, negate=False):
#     def __filter_by_rel(n, check_subtrees=True):
#         result = []
#         rel_match = n.get_rel() == value
#         if (rel_match and (not negate)) or ((not rel_match) and negate):
#             result.append(n)
        
#         if check_subtrees:
#             # TOFIX: selectByPos returns udon2.NodeList, but want just a python list
#             if negate:
#                 result.extend(n.select_except_rel(value)) # search in the subtree
#             else:
#                 result.extend(n.select_by_rel(value)) # search in the subtree 
#         return result

#     if type(node) == list:
#         # see comment in filter_by_morph
#         res = []
#         for nd in node:
#             res.extend(__filter_by_rel(nd, check_subtrees=False))
#     else:
#         res = __filter_by_rel(node)
#     return res

def filter_by_rel_chain(node, rel_chain, check_subtrees=True):
    # TODO: make negate work
    cand, res = [(node, node)], []
    for i, r in enumerate(rel_chain):
        for beg, end in cand:
            if i == 0 and check_subtrees:
                for nd in end.select_by("deprel", str(r)): # search in the subtree
                    res.append((nd.parent, nd))
            else:
                for nd in end.get_by("deprel", str(r)): # not search in the subtree
                    res.append((beg, nd))
        cand, res = res, []
    return cand


def check_arity(node, value, arity, check_subtrees=False):
    freq, cand, node2id = defaultdict(int), {}, {}
    idx = 0
    for beg, end in filter_by_rel_chain(node, value, check_subtrees=check_subtrees):
        el_id = node2id.get(str(beg))
        if el_id is None:
            node2id[str(beg)] = idx
            el_id = idx
            idx += 1
        cand[el_id] = beg
        freq[el_id] += 1
    return [cand[idx] for idx, f in freq.items() if ((check_subtrees and cand[idx] == node) or not check_subtrees) and f >= arity]


def filter_by_morph(node, value, f=None, check_subtrees=False, negate=False):
    def __filter_by_morph(n):
        result = []
        # value is MorphKeyValChain
        v = str(value)

        node_has_all_morph = n.has_all("feats", v)
        if (node_has_all_morph and (not negate)) or ((not node_has_all_morph) and negate):
            result.append(n)

        if f is not None and identify_fn:
            # if we don't have any tags before we might possibly want to check the subtrees
            # otherwise they have been checked by `filter_by_rel_chain`, so all that's left 
            # is to check POS-tags on the nodes themselves, which we have already done above.
            if check_subtrees:
                result.extend(n.select_having("feats", v, negate)) # search in the subtree
        return result

    identify_fn = True
    if f is not None:
        identity_fn = f(node, check_subtrees) == node
        node = f(node, check_subtrees)
    
    if type(node) == list:
        # this means we've invoked a chain before, e.g.:
        # w.acl.nsubj.morph has_not PronType=Rel
        # This will select all w.acl.nsubj and then we efectively will want just to check those nodes
        # for the given condition and not the whole subtrees.
        res = []
        for beg, end in node:
            filtered = __filter_by_morph(end)
            if filtered:
                res.append(beg)

    else:
        filtered = __filter_by_morph(node)
        res = [node] if filtered else None
    return res


def relation_exists(node, value, check_subtrees=False):
    if len(value) == 1:
        if check_subtrees:
            return [x.parent for x in node.select_by("deprel", str(value[0]))]
        else:
            if node.child_has_prop("deprel", str(value[0])):
                return [node]
            else:
                return []
    else:
        res = filter_by_rel_chain(node, value, check_subtrees=check_subtrees)
        # TODO: fix this weird if statement
        return [beg for beg, end in res if (check_subtrees and beg == node) or not check_subtrees]


def relation_not_exists(node, value, check_subtrees=False):
    nodes = relation_exists(node, value, check_subtrees)

    if check_subtrees:
        return [x for x in node.get_subtree_nodes() if x not in nodes]
    else:

        return [] if nodes else [node]


def prop2val(node, prop):
    if prop == 'lemma':
        return node.lemma
    elif prop == 'text':
        return node.form
    elif prop == 'pos':
        return node.upos
    elif prop == 'morph':
        return str(node.feats)
    elif prop == 'rel':
        return node.deprel


class TemplateInstance:
    def __init__(self, cand, text=[]):
        self.__chain = chain_to_root(cand, only_ids=True) if cand else None
        self.__text = set()
        for t in text:
            self.__text.add(t)

    def add_text(self, text):
        self.__text.add(text)

    @property
    def text(self):
        return self.__text

    @property
    def chain(self):
        return self.__chain

    @property
    def is_empty(self):
        return bool(self.__text)


def placeholder2lambda(node, pholder):
    def node2text(n):
        if type(pholder) == NodePlaceholder:
            return prop2val(n, pholder.prop) if hasattr(pholder, 'prop') else n.form
        elif type(pholder) == SubtreePlaceholder:
            return n.get_subtree_text()
        else:
            return None


    if pholder.name == 'qw':
        # this is a question word, we leave it like that
        el = TemplateInstance(None)
        el.add_text(pholder.name)
        return {None: el}
    elif hasattr(pholder, 'tag') and pholder.tag:
        node.reset_subtree()
        candidates = node.get_by_deprel_chain(".".join([str(tag) for tag in pholder.tag]))
        last_tag = pholder.tag[-1]

        # print(".".join([str(tag) for tag in pholder.tag]))
        # print("PHOLDER", pholder, type(pholder), len(candidates), "<--", node.form, " (", node.get_subtree_text(), ")")

        if candidates:
            fixed_cand = {}
            for cand in candidates:
                cand.reset_subtree()
                if hasattr(pholder, 'neg'):
                    if len(pholder.neg) > 0:
                        options = []
                        # make sure the exact number of things is being ignored, i.e. if there are two advcls in pholder.neg and
                        # two of them can be found, just ignore them right away without adding them to options.

                        to_ignore = {}
                        for t in pholder.neg:
                            key = str(t)
                            if key not in to_ignore:
                                to_ignore[key] = [1, t]
                            else:
                                to_ignore[key][0] += 1

                        for str_chain, value in to_ignore.items():
                            repeats, t = value

                            if len(t.chain) > 0:
                                last_el = t.chain[-1]
                            else:
                                last_el = t.tag

                            if last_el.exact:
                                # this is to correctly parse templates like <w.advcl#8 - advcl#8* - advcbl#9>
                                # One should understand that if t is `advcl#8*`, then the same node
                                # as the current template root should simply be ignored
                                if last_tag.id == last_el.id:
                                    cand.ignore()
                                    continue

                                res_by_chain = cand.get_by_deprel_chain(str_chain[:-1])
                                N = len(res_by_chain)
                                
                                if N > 0:
                                    if N <= repeats:
                                        for r_by_chain in res_by_chain:
                                            r_by_chain.ignore()
                                    else:
                                        # we have more results than we need to remove
                                        options.append([(combo, False) for combo in combinations(res_by_chain, repeats)])
                                elif cand.deprel == str_chain[:-1]:
                                    cand.ignore()
                            else:
                                res_by_chain = cand.get_by_deprel_chain(str_chain)
                                N = len(res_by_chain)
                                if N > 0:
                                    if N <= repeats:
                                        for r_by_chain in res_by_chain:
                                            r_by_chain.ignore_subtree()
                                    else:
                                        # we have more results than we need to remove
                                        options.append([(combo, True) for combo in combinations(res_by_chain, repeats)])
                                elif cand.deprel == str_chain:
                                    cand.ignore_subtree()

                        if options:
                            cand_idx, cand_limit = 0, 50
                            fixed_cand[cand.id] = TemplateInstance(cand)
                            for option in product(*options):
                                cand_idx += 1
                                if cand_idx >= cand_limit: break # computational simplification
                                for nodes, subtree in option:
                                    for node in nodes:
                                        if subtree:
                                            node.ignore_subtree()
                                        else:
                                            node.ignore()
                                fixed_cand[cand.id].add_text(node2text(cand))
                                for nodes, subtree in option:
                                    for node in nodes:
                                        if subtree:
                                            node.reset_subtree()
                                        else:
                                            node.reset()
                if cand.id not in fixed_cand:
                    fixed_cand[cand.id] = TemplateInstance(cand)
                    fixed_cand[cand.id].add_text(node2text(cand))
            return fixed_cand
        else:
            # many of these occur for some reason!
            return None
    elif pholder.name == 'w':
        el = TemplateInstance(node)
        if type(pholder) == NodePlaceholder:
            el.add_text(prop2val(node, pholder.prop) if hasattr(pholder, 'prop') else node.form)
            return {node.id: el}
        elif type(pholder) == SubtreePlaceholder:
            el.add_text(node.get_subtree_text())
            return {node.id: el}
        else:
            print("HERE3")
            return None
    else:
        el = TemplateInstance(None)
        el.add_text(str(pholder.name))
        return {None: el}


class SharedElement:
    def __init__(self, q_id, a_id, q_last, a_last):
        self.__q_id = q_id
        self.__a_id = a_id
        self.__last_in_chain = {
            'q': q_last,
            'a': a_last
        }

    @property
    def q(self):
        return self.__q_id

    @property
    def a(self):
        return self.__a_id

    def last_in_chain(self, k):
        return self.__last_in_chain.get(k, None)

    def __str__(self):
        return f"q{self.__q_id}:{self.__last_in_chain['q']}/a{self.__a_id}:{self.__last_in_chain['a']}"


def load_templates(files):
    templates = {}
    for fname in files:
        name_without_ext = os.path.splitext(os.path.basename(fname))[0]
        with open(fname, encoding='utf8') as f:
            for i, temp in enumerate(f):
                template = parse(temp.strip(), Template)
                tags = defaultdict(int)

                answer_ids = {
                    t.id: {'id': i, 'last': j == len(tmpl.tag) - 1}
                    for i, tmpl in enumerate(template.answer)
                    for j, t in enumerate(tmpl.tag if type(tmpl) != str else []) if hasattr(t, 'id')
                }

                templates[f'{name_without_ext}{i+1}'.strip()] = {
                    'wireframe': [type(x) for x in template],
                    'question': lambda n,t=template: [placeholder2lambda(n, x)
                        if type(x) in [SubtreePlaceholder, NodePlaceholder] else {None: TemplateInstance(None, [x])} for x in t],
                    'answer': lambda n,v=template.answer: [placeholder2lambda(n, x)
                        if type(x) in [SubtreePlaceholder, NodePlaceholder] else {None: TemplateInstance(None, [x])} for x in v],
                    'template': temp.strip(),
                    'shared_tags': [
                        SharedElement(i, answer_ids[t.id]['id'], j == len(tmpl.tag) - 1, answer_ids[t.id]['last']) 
                        for i, tmpl in enumerate(template)
                        for j, t in enumerate(tmpl.tag if type(tmpl) != str else []) if hasattr(t, 'id') and t.id in answer_ids
                    ]
                }
    return templates


def load_template_examples(files):
    examples = {}
    for fname in files:
        with open(fname) as f:
            for line in f:
                if line.startswith("id:"):
                    _, temp_id = line.split(": ")
                    examples[temp_id.strip()] = []
                elif line.strip():
                    s, q, a = line.strip().split(" | ")
                    examples[temp_id.strip()].append({
                        "sentence": s,
                        "question": q,
                        "answer": a
                    })
    return examples


def parse_template(template):
    return parse(template.strip(), Template)


def parse_guard(guard):
    return parse(guard.strip(), Guard)