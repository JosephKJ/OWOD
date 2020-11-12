#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52125' --resume --config-file ./configs/OWOD/t2/t2_train_baseline_only_frcnn.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#cp -r /home/fk1/workspace/OWOD/output/t2_baseline /home/fk1/workspace/OWOD/output/t2_ft_baseline
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t2/t2_ft.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#cp -r /home/fk1/workspace/OWOD/output/t2_ft_baseline /home/fk1/workspace/OWOD/output/t3_baseline
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t3/t3_train_baseline_only_frcnn.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --resume --config-file ./configs/OWOD/iOD/10_p_10/next_10_train_with_unk_det.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --resume --config-file ./configs/OWOD/iOD/10_p_10/ft.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52134' --resume --config-file ./configs/OWOD/iOD/10_p_10/ft_with_unk.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
