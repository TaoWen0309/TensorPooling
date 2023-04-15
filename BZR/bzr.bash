for i in {0..9}
do
  python wit_topopool_main_chemical_compound.py --epochs 100 --batch_size 32 --lr 0.01 --fold_idx $i --hidden_dim 16 --tensor_layer_type TCL
done