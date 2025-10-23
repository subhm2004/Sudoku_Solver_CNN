# Sudoku Solver 🤖🧩 

This project is an automated Sudoku solver that takes a screenshot of a Sudoku puzzle, extracts the digits, solves the puzzle, and then automatically fills in the solution on the screen. It leverages image processing techniques, a Convolutional Neural Network (CNN) for digit recognition, and a constraint satisfaction algorithm to solve the puzzle.

🚀 **Key Features**

*   📸 **Screenshot Capture:** Automatically captures a screenshot of the Sudoku puzzle.
*   👁️ **Board Detection:** Detects the Sudoku board within the screenshot.
*   🔢 **Digit Recognition:** Employs a CNN to recognize the digits in each cell.
*   🧠 **Sudoku Solving:** Solves the Sudoku puzzle using a constraint satisfaction algorithm.
*   ✍️ **Automated Filling:** Automatically fills the solved puzzle on the screen.
*   ⏱️ **Performance Tracking:** Measures and reports the time taken for each stage of the process.

🛠️ **Tech Stack**

*   **Programming Language:** Python
*   **Image Processing:** OpenCV (`cv2`)
*   **Numerical Computation:** NumPy
*   **Deep Learning:** PyTorch (`torch`)
*   **GUI Automation:** PyAutoGUI
*   **Image Handling:** Pillow (`PIL.Image`)
*   **Other:** `math`, `itertools`, `time`, `sys`

📦 **Getting Started**

### Prerequisites

*   Python 3.x
*   Pip package manager
*   A Sudoku game running on your screen

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/subhm2004/Sudoku_Solver_CNN.git
    cd Sudoku_Solver_CNN
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running Locally

1.  Ensure a Sudoku game is visible on your screen.
2.  Run the `main.py` script:

    ```bash
    python main.py
    ```

3.  The script will prompt you to press Enter. After pressing Enter, quickly switch to the Sudoku game window. The script will automatically take a screenshot, solve the puzzle, and fill in the solution.

📂 **Project Structure**

```
sudoku-solver/
├── CNN/
│   └── digitsCNN.py       # CNN model definition for digit recognition
├── readBoard.py           # Image processing and digit recognition logic
├── sudoku.py              # Sudoku solving logic
├── main.py                # Main entry point of the application
└── README.md              # This file
```

📝 **License**

This project is licensed under the [MIT License](LICENSE).

📬 **Contact**

If you have any questions or suggestions, please feel free to contact me at [subhu04012003@gmail.com](mailto:subhu04012003@gmail.com).

