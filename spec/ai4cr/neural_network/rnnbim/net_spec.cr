require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rnnbim::Net do
  describe "with default params" do
    net = Ai4cr::NeuralNetwork::Rnnbim::Net.new

    expected_sub_keys_for_hidden_weights = [:past, :local, :future, :combo]

    using_default_params_expected_input_state_qty = 4
    using_default_params_expected_output_state_qty = 2
    using_default_params_expected_hidden_layer_qty = 2
    using_default_params_expected_time_column_qty = 2
    using_default_params_expected_hidden_layer_scale = 1.0
    using_default_params_expected_hidden_state_qty = 3 # (avg of in & out)*scale

    using_default_params_expected_time_column_range = (0..7)
    using_default_params_expected_input_state_range = (0..3) # (0..using_default_params_expected_input_state_qty-1)
    using_default_params_expected_hidden_state_range = (0..2) # (0..using_default_params_expected_hidden_state_qty-1)
    using_default_params_expected_output_state_range = (0..1) # (0..using_default_params_expected_output_state_qty-1)

    using_default_params_expected_hidden_delta_scales = [1,2]

    using_default_params_expected_nodes_hidden = [
      {
        :current => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        },
        :mem_same_image => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        },
        :mem_after_image => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        }
      },
      {
        :current => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        },
        :mem_same_image => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        },
        :mem_after_image => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        }
    }
    ]
    
    using_default_params_expected_weight_meta = {
      "output" => {
        :output => {
          :size => using_default_params_expected_time_column_range.size,
          :time_col_first_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          }
        }
      },
      "hidden_0" => {
        :past => {
          :size => using_default_params_expected_time_column_range.size,
          :time_col_first_keys => {
            :from_inputs => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :from_inputs => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :from_inputs => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          }
        },

        :local => {
          :size => using_default_params_expected_time_column_range.size,
          :time_col_first_keys => {
            :from_inputs_current => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_inputs_future => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :from_inputs_past => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_inputs_current => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_inputs_future => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :from_inputs_past => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_inputs_current => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          }
        },

        :future => {
          :size => using_default_params_expected_time_column_range.size,
          :time_col_first_keys => {
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_inputs => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_inputs => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :from_inputs => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          }
        },

        :combo => {
          :size => using_default_params_expected_time_column_range.size,
          :time_col_first_keys => {
            :from_inputs => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_local => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :from_inputs => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_local => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :from_inputs => {
              :size => using_default_params_expected_input_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_local => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          }
        },
      },
      "hidden_1" => {
        :past => {
          :size => using_default_params_expected_time_column_range.size,
          :time_col_first_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          }
        },

        :local => {
          :size => using_default_params_expected_time_column_range.size,
          :time_col_first_keys => {
            :from_combo_current => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_combo_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :from_combo_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_combo_current => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_combo_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :from_combo_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_combo_current => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          }
        },

        :future => {
          :size => using_default_params_expected_time_column_range.size,
          :time_col_first_keys => {
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          }
        },

        :combo => {
          :size => using_default_params_expected_time_column_range.size,
          :time_col_first_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_local => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_local => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :from_combo => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_past => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_local => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_future => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_same_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_mem_after_image => {
              :size => using_default_params_expected_hidden_state_range.size,
              :sub_size => using_default_params_expected_hidden_state_range.size
            },
            :from_bias => {
              :size => 1,
              :sub_size => using_default_params_expected_hidden_state_range.size
            }
          }
        },
      }
    }
    
    describe "#initialize" do
      
      describe "sets expected quantity for instance variable" do
        it "@input_state_qty" do
          net.input_state_qty.should eq(using_default_params_expected_input_state_qty)
        end
  
        it "@hidden_state_qty" do
          net.hidden_state_qty.should eq(using_default_params_expected_hidden_state_qty)
        end
  
        it "@output_state_qty" do
          net.output_state_qty.should eq(using_default_params_expected_output_state_qty)
        end
  
        it "@hidden_layer_qty" do
          net.hidden_layer_qty.should eq(using_default_params_expected_hidden_layer_qty)
        end
  
        it "@hidden_layer_scale" do
          net.hidden_layer_scale.should eq(using_default_params_expected_hidden_layer_scale)
        end
  
        it "@hidden_delta_scales" do
          net.hidden_delta_scales.should eq(using_default_params_expected_hidden_delta_scales)
        end
      end

      describe "sets expected range for instance variable" do
        it "@input_state_range" do
          net.input_state_range.should eq(using_default_params_expected_input_state_range)
        end
  
        it "@hidden_state_range" do
          net.hidden_state_range.should eq(using_default_params_expected_hidden_state_range)
        end
  
        it "@output_state_range" do
          net.output_state_range.should eq(using_default_params_expected_output_state_range)
        end
      end

      describe "sets expected array structure for" do
        using_default_params_expected_nodes_in = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        using_default_params_expected_nodes_out = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
  
        it "@nodes_in" do
          net.nodes_in.should eq(using_default_params_expected_nodes_in)
        end
          
        it "#nodes_out" do
          net.nodes_out.should eq(using_default_params_expected_nodes_out)
        end
      end

      describe "sets expected hash structure for" do
        it "@nodes_hidden" do
          net.nodes_hidden.should eq(using_default_params_expected_nodes_hidden)
        end
      end
    end

    describe "#init_nodes_hidden" do
      it "returns expected results" do
        net.init_nodes_hidden.should eq(using_default_params_expected_nodes_hidden)
      end
    end

    describe "#init_weights" do
      weights = net.init_weights

      it "has expected top level keys" do
        weights.keys.should eq(using_default_params_expected_weight_meta.keys)
      end

      using_default_params_expected_weight_meta.keys.each do |top_level_key|
        describe "for top level key #{top_level_key}" do
          it "has expected sub-keys" do
            weights[top_level_key].keys.should eq(using_default_params_expected_weight_meta[top_level_key].keys)
          end

          using_default_params_expected_weight_meta[top_level_key].each do |sub_key, sub_value|
            describe "for sub-key #{sub_key} has an array" do
              it "of expected size" do
                weights[top_level_key][sub_key].size.should eq(using_default_params_expected_weight_meta[top_level_key][sub_key][:size])
              end

              describe "with a first element which" do
                it "has expected keys" do
                  # weights[top_level_key][sub_key].first.should eq("TBD")
                  keys = weights[top_level_key][sub_key].first.keys.sort
                  expected_keys = using_default_params_expected_weight_meta[top_level_key][sub_key][:time_col_first_keys].as(Hash(Symbol, Hash(Symbol, Int32))).keys.sort
                  keys.should eq(expected_keys) # [:time_col_first_keys])
                end

                # describe "has expected quantity of values" do
                #   expected_keys = using_default_params_expected_weight_meta[top_level_key][sub_key]

                #   expected_keys.each do |exp_key|
                #     it "has expected quantity of values" do
                #       # weights[top_level_key][sub_key].first.should eq("TBD")
                #       values = weights[top_level_key][sub_key].first[exp_key]
                #       expected_values_size = using_default_params_expected_weight_meta[top_level_key][sub_key][:time_col_first_keys].as(Hash(Symbol, Hash(Symbol, Int32)))[exp_key][:size]
                #       values.size.should eq(expected_values_size) # [:time_col_first_keys])
                #     end
                #   end
                # end
              end

              describe "with a last element which" do
                it "has expected keys" do
                  # weights[top_level_key][sub_key].last.should eq("TBD")
                  keys = weights[top_level_key][sub_key].last.keys.sort
                  expected_keys = using_default_params_expected_weight_meta[top_level_key][sub_key][:time_col_last_keys].as(Hash(Symbol, Hash(Symbol, Int32))).keys.sort
                  keys.should eq(expected_keys) # [:time_col_last_keys])
                end
              end

            end
          end

        end
      end

      ####

      it "returns expected top level keys" do
        top_keys = weights.keys
        expected_keys = ["output"]
        (0..using_default_params_expected_hidden_layer_qty - 1).each { |i| expected_keys << "hidden_#{i}" }
        top_keys.should eq(expected_keys)
      end
    
      it "has expected quantity of top level keys" do
        keys = weights.keys
        expected_key_qty = 1 + net.hidden_layer_range.size
        keys.size.should eq(expected_key_qty)
      end

      describe "has top level key" do
        expected_subkey_class = Hash(Symbol, Array(Hash(Symbol, Array(Array(Float64)))))

        
        describe "output" do
          it "with expected sub-keys" do
            keys = weights["output"].keys
            expected_sub_keys_for_output = [:output] # TODO
            keys.should eq(expected_sub_keys_for_output)
          end

          it "with value of expected class" do
            weights["output"].class.should eq(expected_subkey_class)
          end
        end

        (0..using_default_params_expected_hidden_layer_qty - 1).each do |i|
          expected_top_key = "hidden_#{i}"
          describe "#{expected_top_key}" do
            it "with expected sub-keys" do
              keys = weights[expected_top_key].keys
              keys.should eq(expected_sub_keys_for_hidden_weights)
            end

            it "with value of expected class" do
              weights[expected_top_key].class.should eq(expected_subkey_class)
            end
          end
        end
      end

      # it "DEBUG output values" do
      #   weights["output"].should eq([0.0]) # TODO: for debugging; remove before merging to master
      # end

      # it "DEBUG hidden_0 values" do
      #   weights["hidden_0"].should eq([0.0]) # TODO: for debugging; remove before merging to master
      # end

      # it "DEBUG hidden_1 values" do
      #   weights["hidden_1"].should eq([0.0]) # TODO: for debugging; remove before merging to master
      # end
    end
  end
end



# icr
# require "./src/ai4cr/neural_network/rnnbim/net.cr"
# net = Ai4cr::NeuralNetwork::Rnnbim::Net.new
# puts net.pretty_inspect

# puts "net.input_state_range: #{net.input_state_range}"
# puts "net.hidden_state_range: #{net.hidden_state_range}"
# puts "net.output_state_range: #{net.output_state_range}"
# puts "net.nodes_out: #{net.nodes_out}"

# time_column_index = 0
# hidden_layer_index = 0
# w = net.init_weights_to_current_past(time_column_index, hidden_layer_index)

# w = net.init_weights_to_current_future(time_column_index, hidden_layer_index)

# w = net.init_weights

# puts "w: #{w.pretty_inspect}"
          
# /home/drhuffman/_dev_/github.com/drhuffman12/ai4cr_alt/src/ai4cr/neural_network/rnnbim/net.cr    

# mkdir spec/ai4cr/neural_network/rnnbim/net

# crystal spec spec/ai4cr/neural_network/rnnbim/net_spec.rb


