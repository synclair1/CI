# Code by: Haris Karim Ladhani & Synclair Saqib Samson
# To create the sudoku puzzle, the following article was used https://medium.com/codex/building-a-sudoku-solver-and-generator-in-python-1-3-f29d3ede6b23
import random
import copy

class Board:
    def __init__(self, code=None):
        self.resetBoard()
        if code:
            self.code = code
            for row in range(9):
                for col in range(9):
                    self.board[row][col] = int(code[0])
                    code = code[1:]
        else:
            self.code = None

    def generateBoard(self, difficulty):
        self.board, solutionBoard = self.boardDifficulty(self.randomBoard(), difficulty)
        return self.codedBoard(), self.codedBoard(solutionBoard)

    def resetBoard(self):
        self.board = [[0] * 9 for _ in range(9)]

    def codedBoard(self, board=None):
        if board:
            return ''.join([str(i) for j in board for i in j])
        else:
            return ''.join([str(i) for j in self.board for i in j])

    def boardDifficulty(self, newBoard, difficulty):
        x = 0
        y = 2
        for i in range(3):
            counter = 0
            while counter < 4:
                row = random.randint(x, y)
                col = random.randint(x, y)
                if self.board[row][col] != 0:
                    self.board[row][col] = 0
                    counter += 1
            x += 3
            y += 3
        
        self.board = copy.deepcopy(newBoard)
        if difficulty == "easy":
            board_zeros = 36
        elif difficulty == "medium":
            board_zeros = 44
        elif difficulty == "hard":
            board_zeros = 52
        elif difficulty == "expert":
            board_zeros = 60
        else:
            return

        counter = 0
        while counter < (board_zeros - 12):
            row = random.randint(0,8)
            col = random.randint(0,8)
            if self.board[row][col] != 0:
                n = self.board[row][col]
                self.board[row][col] = 0

                if len(self.findSolutions()) != 1:
                    self.board[row][col] = n
                    continue
                counter += 1
        return self.board, newBoard

    def randomBoard(self):
        self.resetBoard()
        r1 = 0
        r2 = 3
        for x in range(3):
            _l = list(range(1, 10))
            for row in range(r1, r2):
                for col in range(r1, r2):
                    _num = random.choice(_l)
                    self.board[row][col] = _num
                    _l.remove(_num)
            r1 += 3
            r2 += 3

        return self.generateCont()
    
    def generateCont(self):
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                if self.board[row][col] == 0:
                    _num = random.randint(1, 9)

                    if self.checkSpace(_num, (row, col)):
                        self.board[row][col] = _num

                        if self.solve():
                            self.generateCont()
                            return self.board

                        self.board[row][col] = 0
        return False
    
    def checkSpace(self, num, space):
        if not self.board[space[0]][space[1]] == 0:
            return False
            
        for col in self.board[space[0]]:
            if col == num:
                return False
        for row in range(len(self.board)):
            if self.board[row][space[1]] == num:
                return False

        for i in range(3):
            for j in range(3):
                if self.board[(space[0] // 3 * 3) + i][(space[1] // 3 * 3) + j] == num:
                    return False
        return True

    def findSolutions(self):
        z = 0
        solutions = []
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                if self.board[row][col] == 0:
                    z += 1
        for i in range(1, z+1):
            boardCopy = copy.deepcopy(self)
            row, col = self.findSpaceSolution(boardCopy.board, i)
            boardSolution = boardCopy.solveNumberSolution(row, col)
            solutions.append(self.codedBoard(boardSolution))
        return list(set(solutions))

    def findSpaceSolution(self, board, num):
        k = 1
        for row in range(len(board)):
            for col in range(len(board[row])):
                if board[row][col] == 0:
                    if k == num:
                        return (row, col)
                    k += 1
        return False

    def findSpaces(self):
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] == 0:
                    return (row, col)
        return False

    def solve(self):
        availableSpace = self.findSpaces()
        if availableSpace:
            row, col = availableSpace
        else:
            return True
        for i in range(1, 10):
            if self.checkSpace(i, (row, col)):
                self.board[row][col] = i
                if self.solve():
                    return self.board
                self.board[row][col] = 0
        return False

    def solveNumberSolution(self, row, col):
        for n in range(1, 10):
            if self.checkSpace(n, (row, col)):
                self.board[row][col] = n
                if self.solve():
                    return self.board
                self.board[row][col] = 0

        return False