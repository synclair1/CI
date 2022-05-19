# Code by: Haris Karim Ladhani & Synclair Saqib Samson
from puzzleGenerator import Board
from sudoku import Sudoku

if __name__ == '__main__':
    board = Board()
    input = ""
    inputBoard = board.generateBoard("easy")
    print("Board Generated")
    for x in inputBoard[0]:
        input += str(x)
        input += "\n"

    with open('input.txt', 'w') as f:
        f.write(input)
    
    s = Sudoku()
    s.load("input.txt")
    solution = s.solve()
    if(solution):
        s.save("solution.txt", solution)