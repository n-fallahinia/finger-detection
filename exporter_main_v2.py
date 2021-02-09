r"""Tool to export an object detection model for inference.
===============================================
Navid Fallahinia - 12/15/2020
BioRobotics Lab

Prepares an object detection tensorflow graph for inference using model
configuration and a trained checkpoint. Outputs associated checkpoint files,
a SavedModel, and a copy of the model config.

The inference graph contains one of three input nodes depending on the user
specified option.
  * `image_tensor`: Accepts a uint8 4-D tensor of shape [1, None, None, 3]
  * `float_image_tensor`: Accepts a float32 4-D tensor of shape
    [1, None, None, 3]
  * `encoded_image_string_tensor`: Accepts a 1-D string tensor of shape [None]
    containing encoded PNG or JPEG images. Image resolutions are expected to be
    the same if more than 1 image is provided.
  * `tf_example`: Accepts a 1-D string tensor of shape [None] containing
    serialized TFExample protos. Image resolutions are expected to be the same
    if more than 1 image is provided.

and the following output nodes returned by the model.postprocess(..):
  * `num_detections`: Outputs float32 tensors of the form [batch]
      that specifies the number of valid boxes per image in the batch.
  * `detection_boxes`: Outputs float32 tensors of the form
      [batch, num_boxes, 4] containing detected boxes.
  * `detection_scores`: Outputs float32 tensors of the form
      [batch, num_boxes] containing class scores for the detections.
  * `detection_classes`: Outputs float32 tensors of the form
      [batch, num_boxes] containing classes for the detections.


Example Usage:
--------------
python exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_dir path/to/checkpoint \
    --output_directory path/to/exported_model_directory
    --use_side_inputs True/False \
    --side_input_shapes dim_0,dim_1,...dim_a/.../dim_0,dim_1,...,dim_z \
    --side_input_names name_a,name_b,...,name_c \
    --side_input_types type_1,type_2

The expected output would be in the directory
path/to/exported_model_directory (which is created if it does not exist)
holding two subdirectories (corresponding to checkpoint and SavedModel,
respectively) and a copy of the pipeline config.

Config overrides (see the `config_override` flag) are text protobufs
(also of type pipeline_pb2.TrainEvalPipelineConfig) which are used to override
certain fields in the provided pipeline_config_path.  These are useful for
making small changes to the inference graph that differ from the training or
eval config.

Example Usage (in which we change the second stage post-processing score
threshold to be 0.5):

python exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_dir path/to/checkpoint \
    --output_directory path/to/exported_model_directory \
    --config_override " \
            model{ \
              faster_rcnn { \
                second_stage_post_processing { \
                  batch_non_max_suppression { \
                    score_threshold: 0.5 \
                  } \
                } \
              } \
            }"

If side inputs are desired, the following arguments could be appended
(the example below is for Context R-CNN).
   --use_side_inputs True \
   --side_input_shapes 1,2000,2057/1 \
   --side_input_names context_features,valid_context_size \
   --side_input_types tf.float32,tf.int32
"""
from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2

tf.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`, `float_image_tensor`]')
flags.DEFINE_string('pipeline_config_path', 'mobilenet_v2.config','Path to a pipeline_pb2.TrainEvalPipelineConfig config file.')
flags.DEFINE_string('trained_checkpoint_dir', './models/model_1/', 'Path to trained checkpoint directory')
flags.DEFINE_string('output_directory', './inference_graph_1', 'Path to write outputs.')
flags.DEFINE_string('config_override', '',
                    'pipeline_pb2.TrainEvalPipelineConfig '
                    'text proto to override pipeline_config_path.')
flags.DEFINE_boolean('use_side_inputs', False,
                     'If True, uses side inputs as well as image inputs.')
flags.DEFINE_string('side_input_shapes', '',
                    'If use_side_inputs is True, this explicitly sets '
                    'the shape of the side input tensors to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. A `/` denotes a break, starting the shape of '
                    'the next side input tensor. This flag is required if '
                    'using side inputs.')
flags.DEFINE_string('side_input_types', '',
                    'If use_side_inputs is True, this explicitly sets '
                    'the type of the side input tensors. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of types, each of `string`, `integer`, or `float`. '
                    'This flag is required if using side inputs.')
flags.DEFINE_string('side_input_names', '',
                    'If use_side_inputs is True, this explicitly sets '
                    'the names of the side input tensors required by the model '
                    'assuming the names will be a comma-separated list of '
                    'strings. This flag is required if using side inputs.')

flags.mark_flag_as_required('pipeline_config_path')
flags.mark_flag_as_required('trained_checkpoint_dir')
flags.mark_flag_as_required('output_directory')


def main(_):
  # ste the gpu (device:GPU:0) 
  print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    # Restrict TensorFlow to only use the first GPU
      try:
          tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
          tf.config.experimental.set_memory_growth(gpus[0], True)
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
      # Visible devices must be set before GPUs have been initialized
          print(e)

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.io.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(FLAGS.config_override, pipeline_config)
    exporter_lib_v2.export_inference_graph(
      FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_dir,
      FLAGS.output_directory, FLAGS.use_side_inputs, FLAGS.side_input_shapes,
      FLAGS.side_input_types, FLAGS.side_input_names)


if __name__ == '__main__':
  app.run(main)
