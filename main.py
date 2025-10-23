import numpy as np
from PIL import Image
import pyautogui
import time
import readBoard
import sudoku
import sys

if __name__ == "__main__":
    print("Press Enter to start solving...")
    
    # Wait for start (macOS-friendly)
    input()   
    
    # 5-second delay to switch to Sudoku window
    print("Starting in 5 seconds...")
    time.sleep(5)

    # Start timer
    startTime = time.time()
    readBoardTime = startTime
    solvedBoardTime = startTime

    # Take screenshot
    img = pyautogui.screenshot()
    
    # LINUX USERS (optional)
    def screenshot(path="screenshot.png"):
        import os
        os.system(f"spectacle -n -f -b -o {path}")
        return Image.open(path)
    # img = screenshot()

    # Loads the image
    imgNA, thresh = readBoard.loadImage(img)

    # Finds the board
    boardContour = readBoard.findBoard(thresh)

    if boardContour is not None:
        # top-left, top-right, bottom-right, bottom-left
        pts = boardContour.reshape(4, 2)
        boardCorner = (int(pts[0,0]), int(pts[0,1]))

        width  = int(np.linalg.norm(pts[3,0] - pts[0,0]))
        height = int(np.linalg.norm(pts[2,1] - pts[0,1]))

        cellSize = int(((width / 9) + (height / 9)) / 2)

        # Cuts the board
        boardImg = readBoard.warpBoard(imgNA, boardContour)

        # Cuts all cells
        cells = readBoard.splitCells(boardImg)

        # Processes Cells
        processedCells = readBoard.preprocessCells(cells)

        # Loads CNN Model
        model = readBoard.loadModel()

        # Reads the board
        board = readBoard.readBoard(model, processedCells)

        readBoardTime = time.time()

        # Makes it a matrix
        board = sudoku.refactorBoard(board)

        print(f"\nBoard: {board}")

        # Solves the sudoku
        solvedBoard, solvedEmpties = sudoku.solveBoard(board)

        print(f"\nSolved Board: {solvedBoard}")

        solvedBoardTime = time.time()

        isSolved = all(all(cell != 0 for cell in row) for row in solvedBoard)
        if isSolved:
            sudoku.FillBoard(solvedEmpties, boardCorner, cellSize)

    else:
        print("Sudoku board not detected. Make sure it's visible on screen.")
        sys.exit(1)

    # End timer
    endTime = time.time()

    print(f"\nTotal Time: {endTime - startTime:.6f} seconds")
    print(f"Read Board Time: {readBoardTime - startTime:.6f} seconds")
    print(f"Solve Time: {solvedBoardTime - startTime:.6f} seconds")
