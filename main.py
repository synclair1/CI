from puzzleGenerator import Board
from sudoku import Sudoku

if __name__ == '__main__':
    board = Board()
    lst = ""
    board_config = board.generateBoard("easy") # generates a medium level sudoku
    print("Board Generated")
    for x in board_config[0]:
        lst += str(x)
        lst += "\n"

    with open('input.txt', 'w') as f:
        f.write(lst)
    
    s = Sudoku()
    s.load("input.txt")
    solution = s.solve()
    if(solution):
        s.save("solution.txt", solution)