for i in {0..9}
do
  python wit_topopool_main_chemical_compound.py --epochs 200 --batch_size 32 --lr 0.01 --fold_idx $i --hidden_dim 64 --tensor_layer_type TCL
done