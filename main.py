from configs.config import args

def run_train():
    from inverse_model_GRU.train import train
    train(args)

def run_test():
    from inverse_model_GRU.test import test
    test(args)

def run_infer():
    from inverse_model_GRU.infer import infer
    infer(args)

if __name__ == "__main__":
    if args.is_training == 1:
        run_train()
    elif args.is_training == 2:
        run_test()
    elif args.is_training == 3:
        run_infer()
    else:
        raise ValueError("is_training must be 1/2/3")