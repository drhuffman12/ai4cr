module Ai4cr
  module NeuralNetwork
    module Rnnbim # RNN, Bidirectional, Inversable Memory
      # Expected metadata for Rnnbim Net and a kind of spec helper
      class NetMeta
        property time_column_range : Range(Int32, Int32)
        property hidden_state_range : Range(Int32, Int32)
        property input_state_range : Range(Int32, Int32)
        property output_state_range : Range(Int32, Int32)

        def initialize(
            @time_column_range,
            @hidden_state_range,
            @input_state_range,
            @output_state_range
          )
        end

        def weights : MetaWeightsNetwork
          {
            "output" => weights_output,
            "hidden_0" => weights_hidden_0,
            "hidden_1" => weights_hidden_other,
          }
        end

        def weights_output()
          {
            :output => { # 
              :chrono_size => time_column_range.size,
              :time_col_first_keys => {
                :combo => {
                  :in_size => hidden_state_range.size, # MetaWeightsSimple
                  :out_size => output_state_range.size
                }, # MetaWeightsFromChannel
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => output_state_range.size
                },
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => output_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => output_state_range.size
                }
              }, # MetaWeightsAtTime
              :time_col_mid_keys => {
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => output_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => output_state_range.size
                },
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => output_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => output_state_range.size
                }
              },
              :time_col_last_keys => {
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => output_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => output_state_range.size
                },
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => output_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => output_state_range.size
                }
              }
            } # MetaWeightsToChannel
          } # MetaWeightsNetwork
        end

        def weights_hidden_0
          {
            :past => {
              :chrono_size => time_column_range.size,
              :time_col_first_keys => {
                :input => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_mid_keys => {
                :input => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_last_keys => {
                :input => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              }
            },
    
            :local => {
              :chrono_size => time_column_range.size,
              :time_col_first_keys => {
                :input_current => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :input_future => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_mid_keys => {
                :input_past => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :input_current => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :input_future => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_last_keys => {
                :input_past => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :input_current => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              }
            },
    
            :future => {
              :chrono_size => time_column_range.size,
              :time_col_first_keys => {
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :input => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_mid_keys => {
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :input => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_last_keys => {
                :input => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              }
            },
    
            :combo => {
              :chrono_size => time_column_range.size,
              :time_col_first_keys => {
                :input => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :local => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_mid_keys => {
                :input => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :local => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_last_keys => {
                :input => {
                  :in_size => input_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :local => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              }
            },
          }
        end

        def weights_hidden_other
          {
            :past => {
              :chrono_size => time_column_range.size,
              :time_col_first_keys => {
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_mid_keys => {
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_last_keys => {
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              }
            },
    
            :local => {
              :chrono_size => time_column_range.size,
              :time_col_first_keys => {
                :combo_current => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :combo_future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_mid_keys => {
                :combo_past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :combo_current => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :combo_future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_last_keys => {
                :combo_past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :combo_current => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              }
            },
    
            :future => {
              :chrono_size => time_column_range.size,
              :time_col_first_keys => {
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_mid_keys => {
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_last_keys => {
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              }
            },
    
            :combo => {
              :chrono_size => time_column_range.size,
              :time_col_first_keys => {
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :local => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_mid_keys => {
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :local => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              },
              :time_col_last_keys => {
                :combo => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :past => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :local => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :future => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_same_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :mem_after_image => {
                  :in_size => hidden_state_range.size,
                  :out_size => hidden_state_range.size
                },
                :bias => {
                  :in_size => 1,
                  :out_size => hidden_state_range.size
                }
              }
            },
          }
        end
      end
    end
  end
end
