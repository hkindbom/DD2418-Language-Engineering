from pathlib import Path
from parse_dataset import Dataset
import argparse

class Parser: 
    SH, LA, RA = 0,1,2

    def conllu(self, source):
        buffer = []
        for line in source:
            line = line.rstrip()    # strip off the trailing newline
            if not line.startswith("#"):
                if not line:
                    yield buffer
                    buffer = []
                else:
                    columns = line.split("\t")
                    if columns[0].isdigit():    # skip range tokens
                        buffer.append(columns)

    def trees(self, source):
        """
        Reads trees from an input source.

        Args: source: An iterable, such as a file pointer.

        Yields: Triples of the form `words`, `tags`, heads where: `words`
        is the list of words of the tree (including the pseudo-word
        <ROOT> at position 0), `tags` is the list of corresponding
        part-of-speech tags, and `heads` is the list of head indices
        (one head index per word in the tree).
        """
        for rows in self.conllu(source):
            words = ["<ROOT>"] + [row[1] for row in rows]
            tags = ["<ROOT>"] + [row[3] for row in rows]
            tree = [0] + [int(row[6]) for row in rows]
            relations = ["root"] + [row[7] for row in rows]
            yield words, tags, tree, relations


    def step_by_step(self,string) :
        """
        Parses a string and builds a dependency tree. In each step,
        the user needs to input the move to be made.
        """
        w = ("<ROOT> " + string).split()
        i, stack, pred_tree = 0, [], [0]*len(w) # Input configuration
        while True :
            print( "----------------" )
            print( "Buffer: ", w[i:] )
            print( "Stack: ", [w[s] for s in stack] )
            print( "Predicted tree: ", pred_tree )
            try :
                m = int(input( "Move (SH=0, LA=1, RA=2): " ))
                if m not in self.valid_moves(i,stack,pred_tree) :
                    print( "Illegal move" )
                    continue
            except :
                print( "Illegal move" )
                continue
            i, stack, pred_tree = self.move(i,stack,pred_tree,m)
            if i == len(w) and stack == [0] :
                # Terminal configuration
                print( "----------------" )
                print( "Final predicted tree: ", pred_tree )
                return

    def create_dataset(self, source) :
        """
        Creates a dataset from all parser configurations encountered
        during parsing of the training dataset.
        (Not used in assignment 1).
        """
        ds = Dataset()
        with open(filename) as source:
            for w,tags,tree,relations in self.trees(source) : 
                i, stack, pred_tree = 0, [], [0]*len(tree) # Input configuration
                m = self.compute_correct_move(i,stack,pred_tree,tree)
                while m != None :
                    ds.add_datapoint(w,tags,i,stack,m)
                    i,stack,pred_tree = self.move(i,stack,pred_tree,m)
                    m = self.compute_correct_move(i,stack,pred_tree,tree)
        return ds.dataset2arrays()
   


    def valid_moves(self, i, stack, pred_tree):
        """Returns the valid moves for the specified parser
        configuration.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            The list of valid moves for the specified parser
                configuration.
        """
        moves = []

        # YOUR CODE HERE
        # can do a shift as long as buffer contains some unprocessed word (not passed last word in sentence)
        if i < len(pred_tree):
            moves.append(0)
        # if we have at least root + 2 words on stack, we can do a left arc operation
        if len(stack) > 2:
            moves.append(1)
        # can do right arc operation if at least 2 things on the stack
        if len(stack) > 1:
            moves.append(2)

        return moves

        
    def move(self, i, stack, pred_tree, move):
        """
        Executes a single move.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            move: The move that the parser should make.
        
        Returns:
            The new parser configuration, represented as a triple
            containing the index of the new first unprocessed word,
            stack, and partial dependency tree.
        """

        # YOUR CODE HERE
        # if shift
        if move == 0:
            stack.append(i)
            i += 1
        # if left arc operation
        elif move == 1:
            target_word_pos = len(stack)-2
            head_word_pos = len(stack)-1

            pred_tree = self.update_pred_tree(pred_tree, stack, target_word_pos, head_word_pos)

            stack.pop(target_word_pos)
        # if right arc operation
        else:
            target_word_pos = len(stack)-1
            head_word_pos = len(stack)-2
            pred_tree = self.update_pred_tree(pred_tree, stack, target_word_pos, head_word_pos)

            stack.pop(target_word_pos)

        return i, stack, pred_tree

    def update_pred_tree(self,pred_tree, stack, target_word_pos, head_word_pos):
        target_word = stack[target_word_pos]
        head_word = stack[head_word_pos]
        pred_tree[target_word] = head_word
        return pred_tree

    def compute_correct_moves(self, tree):
        """
        Computes the sequence of moves (transformations) the parser 
        must perform in order to produce the input tree.
        """
        i, stack, pred_tree = 0, [], [0]*len(tree) # Input configuration
        moves = []
        m = self.compute_correct_move(i,stack,pred_tree,tree)
        while m != None :
            moves.append(m)
            i,stack,pred_tree = self.move(i,stack,pred_tree,m)
            m = self.compute_correct_move(i,stack,pred_tree,tree)
        return moves


    def compute_correct_move(self, i,stack,pred_tree,correct_tree) :
        """
        Given a parser configuration (i,stack,pred_tree), and 
        the correct final tree, this method computes the  correct 
        move to do in that configuration.
    
        See the textbook, chapter 15, page 11. 
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            correct_tree: The correct dependency tree.
        
        Returns:
            The correct move for the specified parser
            configuration, or `None` if no move is possible.
        """
        assert len(pred_tree) == len(correct_tree)

        # YOUR CODE HERE
        valid_moves = self.valid_moves(i, stack, pred_tree)
        move = None
        if 1 in valid_moves:
            # Choose LEFTARC if it produces a correct head-dependent relation given the
            # reference parse and the current configuration
            LA_target_word_pos = len(stack) - 2
            LA_head_word_pos = len(stack) - 1
            LA_target_word = stack[LA_target_word_pos]
            LA_head_word = stack[LA_head_word_pos]

            if correct_tree[LA_target_word] == LA_head_word:
                move = 1

        if 2 in valid_moves and move is None:
            # Otherwise, choose RIGHTARC if (1) it produces a correct head-dependent
            # relation given the reference parse and (2) all of the dependents of the word at
            # the top of the stack have already been assigned
            RA_target_word_pos = len(stack) - 1
            RA_head_word_pos = len(stack) - 2
            RA_target_word = stack[RA_target_word_pos]
            RA_head_word = stack[RA_head_word_pos]

            if correct_tree[RA_target_word] == RA_head_word and self.all_dep_made(RA_target_word, pred_tree, correct_tree):
                move = 2

        # Otherwise, choose SHIFT.
        if 0 in valid_moves and move is None:
            move = 0

        return move
   
    def all_dep_made(self, target_word, pred_tree, correct_tree):
        return True if pred_tree.count(target_word) == correct_tree.count(target_word) else False

  
filename = Path("en-ud-train-projective.conllu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transition-based dependency parser')
    parser.add_argument('-s', '--step_by_step', type=str, help='step-by-step parsing of a string')
    parser.add_argument('-m', '--compute_correct_moves', type=str, default=filename, help='compute the correct moves given a correct tree')
    args = parser.parse_args()

    p = Parser()
    if args.step_by_step:
        p.step_by_step( args.step_by_step )

    elif args.compute_correct_moves:
        with open(filename) as source:
            for w,tags,tree,relations in p.trees(source):
                print( p.compute_correct_moves(tree) )







