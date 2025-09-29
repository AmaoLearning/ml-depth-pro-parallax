# x y = 500 300
depth-pro-run -i ./data/example.jpg -o ./outputs/
python ./src/parallax_vanilla/main.py -i ./data/example.jpg -d ./outputs/example.npz -o ./outputs/ -g 2
python ./src/parallax_vanilla/main.py -i ./data/example.jpg -d ./outputs/example.npz -o ./outputs/ -g 1
