<<<<<<< HEAD
@echo off

echo Opening VS Code...
start "" "D:\Alzheimer_Project"

echo Starting Flask Server...
start cmd /k "cd /d D:\Alzheimer_Project && call env\Scripts\activate && python app.py"

timeout /t 5

echo Opening Browser...
start http://127.0.0.1:5000

=======
@echo off

echo Opening VS Code...
start "" "D:\Alzheimer_Project"

echo Starting Flask Server...
start cmd /k "cd /d D:\Alzheimer_Project && call env\Scripts\activate && python app.py"

timeout /t 5

echo Opening Browser...
start http://127.0.0.1:5000

>>>>>>> f3d555db94834b24b87a0a8fc17ffcabab2f8324
pause