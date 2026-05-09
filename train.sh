
python train.py --dataset COTTON --seed 3405
python train.py --dataset COTTON --seed 3407
python train.py --dataset COTTON --seed 3409

python train.py --dataset soybean200 --seed 3405
python train.py --dataset soybean200 --seed 3407
python train.py --dataset soybean200 --seed 3409


python train.py --dataset SoyAgeing --seed 3407 --stage R1
python train.py --dataset SoyAgeing --seed 3407 --stage R3
python train.py --dataset SoyAgeing --seed 3407 --stage R4
python train.py --dataset SoyAgeing --seed 3407 --stage R5
python train.py --dataset SoyAgeing --seed 3407 --stage R6


python train.py --dataset SoyCultivar200 --seed 3407 --position L
python train.py --dataset SoyCultivar200 --seed 3407 --position L  --swap True
python train.py --dataset SoyCultivar200 --seed 3407 --position M
python train.py --dataset SoyCultivar200 --seed 3407 --position M  --swap True
python train.py --dataset SoyCultivar200 --seed 3407 --position U
python train.py --dataset SoyCultivar200 --seed 3407 --position U  --swap True
python train.py --dataset SoyCultivar200 --seed 3407 --position all
python train.py --dataset SoyCultivar200 --seed 3407 --position all  --swap True


python train.py --dataset SoyGene --seed 3407 --stage R6

python train.py --dataset SoyGlobal --seed 3407 --stage R6