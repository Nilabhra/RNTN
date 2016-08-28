import sys

word_to_word_mapping = {
    '{': '-LCB-',
    '}': '-RCB-'
}
word_to_POS_mapping = {
    '--': ':',
    '-': ':',
    ';': ':',
    ':': ':',
    '-LRB-': '-LRB-',
    '-RRB-': '-RRB-',
    '-LCB-': '-LRB-',
    '-RCB-': '-RRB-',
    '{': '-LRB-',
    '}': '-RRB-',
    'Wa': 'NNP'
}
def standardise_node(tree):
    if tree.word in word_to_word_mapping:
        tree.word = word_to_word_mapping[tree.word]
    if tree.word in word_to_POS_mapping:
        tree.label = word_to_POS_mapping[tree.word]

class PTB_Tree:
    '''Tree for PTB format

    >>> tree = PTB_Tree()
    >>> tree.set_by_text("(ROOT (NP (NNP Newspaper)))")
    >>> print tree
    (ROOT (NP (NNP Newspaper)))
    >>> tree = PTB_Tree()
    >>> tree.set_by_text("(ROOT (S (NP-SBJ (NNP Ms.) (NNP Haag) ) (VP (VBZ plays) (NP (NNP Elianti) )) (. .) ))")
    >>> print tree
    (ROOT (S (NP-SBJ (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))
    >>> print tree.word_yield()
    Ms. Haag plays Elianti .
    >>> tree = PTB_Tree()
    >>> tree.set_by_text("(ROOT (NFP ...))")
    >>> print tree
    (ROOT (NFP ...))
    >>> tree.word_yield()
    '...'
    '''
# Convert text from the PTB to a tree. For example:
# ( (S (NP-SBJ (NNP Ms.) (NNP Haag) ) (VP (VBZ plays) (NP (NNP Elianti) )) (. .) ))
# This is a compressed form of:
# ( (S
#     (NP-SBJ (NNP Ms.) (NNP Haag))
#     (VP (VBZ plays)
#       (NP (NNP Elianti)))
#     (. .)))
    def __init__(self):
        self.subtrees = []
        self.word = None
        self.label = ''

    def set_by_text(self, text, pos=0):
        depth = 0
        for i in xrange(pos + 1, len(text)):
            char = text[i]
            # update the depth
            if char == '(':
                depth += 1
                if depth == 1:
                    subtree = PTB_Tree()
                    subtree.set_by_text(text, i)
                    self.subtrees.append(subtree)
            elif char == ')':
                depth -= 1
                if len(self.subtrees) == 0:
                    pos = i
                    for j in xrange(i, 0, -1):
                        if text[j] == ' ':
                            pos = j
                            break
                    self.word = text[pos + 1:i]

            # we've reached the end of the category that is the root of this subtree
            if depth == 0 and char == ' ' and self.label == '':
                self.label = text[pos + 1:i]
            # we've reached the end of the scope for this bracket
            if depth < 0:
                break

        # Fix some issues with variation in output, and one error in the treebank
        # for a word with a punctuation POS
#        standardise_node(self)

    def word_yield(self, span=None, pos=-1):
        return_tuple = True
        if pos < 0:
            pos = 0
            return_tuple = False
        ans = None
        if self.word is not None:
            if span is None or span[0] <= pos < span[1]:
                ans = (pos + 1, self.word)
            else:
                ans = (pos + 1, '')
        else:
            text = []
            for subtree in self.subtrees:
                pos, words = subtree.word_yield(span, pos)
                if words != '':
                    text.append(words)
            ans = (pos, ' '.join(text))
        if return_tuple:
            return ans
        else:
            return ans[1]

    def __repr__(self, single_line=True, depth=0):
        ans = ''
        if not single_line and depth > 0:
            ans = '\n' + depth * '\t'
        ans += '(' + self.label
        if self.word is not None:
            ans += ' ' + self.word
        for subtree in self.subtrees:
            if single_line:
                ans += ' '
            ans += subtree.__repr__(single_line, depth + 1)
        ans += ')'
        return ans


def read_tree(source):
    cur_text = []
    depth = 0
    while True:
        line = source.readline()
        # Check if we are out of input
        if line == '':
            return None
        # strip whitespace and only use if this contains something
        line = line.strip()
        if line == '':
            continue
        cur_text.append(line)
        # Update depth
        for char in line:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
        # At depth 0 we have a complete tree
        if depth == 0:
            tree = PTB_Tree()
            tree.set_by_text(' '.join(cur_text))
            return tree
    return None

def read_trees(source, max_sents=-1):
    if type(source) == type(''):
        source = open(source)
    trees = []
    while True:
        tree = read_tree(source)
        if tree is None:
            break
        trees.append(tree)
        if len(trees) >= max_sents > 0:
            break
    return trees

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage:\n%s <filename>" % sys.argv[0]
        print "Running doctest"
        import doctest
        doctest.testmod()
    else:
        filename = sys.argv[1]
        trees = read_PTB_trees(filename)
        print len(trees), "trees read from", filename
        for tree in trees:
            print tree
