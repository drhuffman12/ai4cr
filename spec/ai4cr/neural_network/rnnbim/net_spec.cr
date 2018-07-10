require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rnnbim::Net do
  describe "with default params" do
    net = Ai4cr::NeuralNetwork::Rnnbim::Net.new

    expected_sub_keys_for_hidden_weights = [:past, :local, :future, :combo]

    default_expected_input_state_qty = 4
    default_expected_output_state_qty = 2
    default_expected_hidden_layer_qty = 2
    default_expected_time_column_qty = 2
    default_expected_hidden_layer_scale = 1.0
    default_expected_hidden_state_qty = 3 # (avg of in & out)*scale

    default_expected_time_column_range = (0..7)
    default_expected_input_state_range = (0..3) # (0..default_expected_input_state_qty-1)
    default_expected_hidden_state_range = (0..2) # (0..default_expected_hidden_state_qty-1)
    default_expected_output_state_range = (0..1) # (0..default_expected_output_state_qty-1)

    default_expected_hidden_delta_scales = [1,2]

    default_expected_nodes_hidden = [
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
    
    ## for Specs:
    # alias ChronoSize = Int32
    # alias LayerSize = Int32
    # alias MetaChronoKey = Symbol
    
    # alias MetaWeightsSimple = Hash(Symbol, ChronoSize)
    # alias MetaWeightsFromChannel = Hash(FromChannelKey, MetaWeightsSimple)
    # alias MetaWeightsAtTime = Hash(MetaChronoKey, MetaWeightsFromChannel)
    # alias MetaWeightsToChannel = Hash(ToChannelKey,MetaWeightsAtTime)
    # alias MetaWeightsNetwork = Hash(LayerName,MetaWeightsToChannel)

    default_expected_weight_meta = {
      "output" => {
        :output => { # 
          :chrono_size => default_expected_time_column_range.size,
          :time_col_first_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size, # MetaWeightsSimple
              :out_size => default_expected_hidden_state_range.size
            }, # Hash(Symbol, Hash(Symbol, Int32))
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }, # MetaWeightsAtTime
          :time_col_mid_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }
        } # Hash(Symbol, Hash(Symbol, Int32))
      }, # MetaWeightsNetwork
      "hidden_0" => {
        :past => {
          :chrono_size => default_expected_time_column_range.size,
          :time_col_first_keys => {
            :input => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :input => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :input => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }
        },

        :local => {
          :chrono_size => default_expected_time_column_range.size,
          :time_col_first_keys => {
            :input_current => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :input_future => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :input_past => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :input_current => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :input_future => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :input_past => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :input_current => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }
        },

        :future => {
          :chrono_size => default_expected_time_column_range.size,
          :time_col_first_keys => {
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :input => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :input => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :input => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }
        },

        :combo => {
          :chrono_size => default_expected_time_column_range.size,
          :time_col_first_keys => {
            :input => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :local => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :input => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :local => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :input => {
              :in_size => default_expected_input_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :local => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }
        },
      },
      "hidden_1" => {
        :past => {
          :chrono_size => default_expected_time_column_range.size,
          :time_col_first_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }
        },

        :local => {
          :chrono_size => default_expected_time_column_range.size,
          :time_col_first_keys => {
            :combo_current => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :combo_future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :combo_past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :combo_current => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :combo_future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :combo_past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :combo_current => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }
        },

        :future => {
          :chrono_size => default_expected_time_column_range.size,
          :time_col_first_keys => {
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }
        },

        :combo => {
          :chrono_size => default_expected_time_column_range.size,
          :time_col_first_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :local => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_mid_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :local => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          },
          :time_col_last_keys => {
            :combo => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :past => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :local => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :future => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_same_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :mem_after_image => {
              :in_size => default_expected_hidden_state_range.size,
              :out_size => default_expected_hidden_state_range.size
            },
            :bias => {
              :in_size => 1,
              :out_size => default_expected_hidden_state_range.size
            }
          }
        },
      }
    }
    
    describe "#initialize" do
      
      describe "sets expected quantity for instance variable" do
        it "@input_state_qty" do
          net.input_state_qty.should eq(default_expected_input_state_qty)
        end
  
        it "@hidden_state_qty" do
          net.hidden_state_qty.should eq(default_expected_hidden_state_qty)
        end
  
        it "@output_state_qty" do
          net.output_state_qty.should eq(default_expected_output_state_qty)
        end
  
        it "@hidden_layer_qty" do
          net.hidden_layer_qty.should eq(default_expected_hidden_layer_qty)
        end
  
        it "@hidden_layer_scale" do
          net.hidden_layer_scale.should eq(default_expected_hidden_layer_scale)
        end
  
        it "@hidden_delta_scales" do
          net.hidden_delta_scales.should eq(default_expected_hidden_delta_scales)
        end
      end

      describe "sets expected range for instance variable" do
        it "@input_state_range" do
          net.input_state_range.should eq(default_expected_input_state_range)
        end
  
        it "@hidden_state_range" do
          net.hidden_state_range.should eq(default_expected_hidden_state_range)
        end
  
        it "@output_state_range" do
          net.output_state_range.should eq(default_expected_output_state_range)
        end
      end

      describe "sets expected array structure for" do
        default_expected_nodes_in = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        default_expected_nodes_out = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
  
        it "@nodes_in" do
          net.nodes_in.should eq(default_expected_nodes_in)
        end
          
        it "#nodes_out" do
          net.nodes_out.should eq(default_expected_nodes_out)
        end
      end

      describe "sets expected hash structure for" do
        it "@nodes_hidden" do
          net.nodes_hidden.should eq(default_expected_nodes_hidden)
        end
      end
    end

    describe "#init_hidden_nodes" do
      it "returns expected results" do
        net.init_hidden_nodes.should eq(default_expected_nodes_hidden)
      end
    end

    describe "#init_network_weights" do
      weights = net.init_network_weights

      it "has expected top level keys" do
        weights.keys.should eq(default_expected_weight_meta.keys)
      end

      default_expected_weight_meta.keys.each do |top_level_key|
        describe "for top level key #{top_level_key}" do
          it "has expected sub-keys" do
            weights[top_level_key].keys.should eq(default_expected_weight_meta[top_level_key].keys)
          end

          default_expected_weight_meta[top_level_key].each do |sub_key, sub_value|
            describe "for sub-key #{sub_key} has an array" do
              it "of expected size" do
                expected_size = sub_value[:chrono_size]
                weights[top_level_key][sub_key].size.should eq(expected_size)
              end

              describe "with a first element which" do
                it "has expected keys" do
                  keys = weights[top_level_key][sub_key].first.keys.sort
                  expected_keys = sub_value[:time_col_first_keys].as(Hash(Symbol, Hash(Symbol, Int32))).keys.sort
                  keys.should eq(expected_keys) # [:time_col_first_keys])
                end

                # describe "has expected quantity of values" do
                #   expected_keys = sub_value

                #   expected_keys.each do |exp_key|
                #     it "has expected quantity of values" do
                #       # weights[top_level_key][sub_key].first.should eq("TBD")
                #       values = weights[top_level_key][sub_key].first[exp_key]
                #       expected_values_size = sub_value[:time_col_first_keys].as(Hash(Symbol, Hash(Symbol, Int32)))[exp_key][:in_size]
                #       values.size.should eq(expected_values_size) # [:time_col_first_keys])
                #     end
                #   end
                # end
              end

              describe "with a last element which" do
                it "has expected keys" do
                  # weights[top_level_key][sub_key].last.should eq("TBD")
                  keys = weights[top_level_key][sub_key].last.keys.sort
                  expected_keys = sub_value[:time_col_last_keys].as(Hash(Symbol, Hash(Symbol, Int32))).keys.sort
                  keys.should eq(expected_keys) # [:time_col_last_keys])
                end
              end

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

# w = net.init_network_weights

# puts "w: #{w.pretty_inspect}"
          
# /home/drhuffman/_dev_/github.com/drhuffman12/ai4cr_alt/src/ai4cr/neural_network/rnnbim/net.cr    

# mkdir spec/ai4cr/neural_network/rnnbim/net

# crystal spec spec/ai4cr/neural_network/rnnbim/net_spec.rb


