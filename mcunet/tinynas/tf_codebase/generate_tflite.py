import tensorflow as tf
import torch
import functools
import numpy as np


def generate_tflite_with_weight(pt_model, resolution, tflite_fname, calib_loader,
                                n_calibrate_sample=500):
    # 1. convert the state_dict to tensorflow format
    pt_sd = pt_model.state_dict()

    tf_sd = {}
    for key, v in pt_sd.items():
        if key.endswith('depth_conv.conv.weight'):
            v = v.permute(2, 3, 0, 1)
        elif key.endswith('conv.weight'):
            v = v.permute(2, 3, 1, 0)
        elif key == 'classifier.linear.weight':
            v = v.permute(1, 0)
        tf_sd[key.replace('.', '/')] = v.numpy()

    # 2. build the tf network using the same config
    weight_decay = 0.

    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session() as sess:
            def network_map(images):
                net_config = pt_model.config
                from mcunet.tinynas.tf_codebase.tf_modules import ProxylessNASNets
                net_tf = ProxylessNASNets(net_config=net_config, net_weights=tf_sd,
                                          n_classes=pt_model.classifier.linear.out_features,
                                          graph=graph, sess=sess, is_training=False,
                                          images=images, img_size=resolution)
                logits = net_tf.logits
                return logits, {}

            def arg_scopes_map(weight_decay=0.):
                arg_scope = tf.contrib.framework.arg_scope
                with arg_scope([]) as sc:
                    return sc

            slim = tf.contrib.slim

            @functools.wraps(network_map)
            def network_fn(images):
                arg_scope = arg_scopes_map(weight_decay=weight_decay)
                with slim.arg_scope(arg_scope):
                    return network_map(images)

            input_shape = [1, resolution, resolution, 3]
            placeholder = tf.compat.v1.placeholder(name='input', dtype=tf.float32, shape=input_shape)

            out, _ = network_fn(placeholder)

            # 3. convert to tflite (with int8 quantization)
            converter = tf.lite.TFLiteConverter.from_session(sess, [placeholder], [out])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.inference_output_type = tf.int8
            converter.inference_input_type = tf.int8

            def representative_dataset_gen():

                for i_b, (data, _) in enumerate(calib_loader):
                    if i_b == n_calibrate_sample:
                        break
                    data = data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
                    yield [data]

            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            tflite_buffer = converter.convert()
            tf.io.gfile.GFile(tflite_fname, "wb").write(tflite_buffer)


if __name__ == '__main__':
    # a simple script to convert the model to TfLite format
    import sys
    sys.path.append('')
    import json

    cfg_path = "cfg_20250607_170204.json"
    ckpt_path = "subnet_20250607_170204.pth"
    tflite_path = "tflite_model.tflite"
    from mcunet.tinynas.nn.networks import MCUNets

    cfg = json.load(open(cfg_path))
    model = MCUNets.build_from_config(cfg)
    if ckpt_path != 'None':
        sd = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(sd)

    # prepare calib loader
    # calibrate the model for quantization
    from torchvision import datasets, transforms
    dataset_mean = [0.23280394, 0.24616548, 0.26092353]
    dataset_std = [0.16994016, 0.17286949, 0.16250615]
    train_dataset = datasets.ImageFolder('data',
                                         transform=transforms.Compose([
                                             # transforms.Resize(int(resolution * 256 / 224)),
                                             # transforms.CenterCrop(resolution),
                                             #transforms.RandomResizedCrop(cfg['resolution']),
                                             transforms.Resize(cfg['resolution']),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=dataset_mean, std=dataset_std)
                                         ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1,
        shuffle=True, num_workers=4)

    generate_tflite_with_weight(model, cfg['resolution'], tflite_path, train_loader,
                                n_calibrate_sample=500)
    print("TFLite model generated at: ", tflite_path)
