# require "./rnn/*"

module Ai4cr
  module NeuralNetwork
    struct Rnn
      enum ChannelType
        Local
        Past
        Future
        Combo
      end
  
      alias NodeCoord = NamedTuple(output_layer: Int32, time_col: Int32, channel: ChannelType)
      alias SimpleNodes = Array(Array(Float64))
  
      property size_time_cols : Int32
      property size_states_in : Int32
      property size_states_out : Int32

      getter range_time_cols : Range(Int32,Int32)
      getter range_states_in : Range(Int32,Int32)
      getter range_states_out : Range(Int32,Int32)

      property inputs : SimpleNodes
      property outputs : SimpleNodes

      

      # property memory : SimpleNodes

      # property hidden_first : HiddenFirst
      # property hidden_others : Array(HiddenOther)


      def initialize(size_time_cols = 5, size_states_in = 3, size_states_out = 4, inputs = nil, outputs = nil) # , hidden_first = nil)
        @size_time_cols = inputs ? inputs.as(SimpleNodes).size : size_time_cols
        @size_states_in = inputs ? inputs.as(SimpleNodes).first.size : size_states_in
        @size_states_out = outputs ? outputs.as(SimpleNodes).first.size : size_states_out

        @range_time_cols = (0..size_time_cols-1)
        @range_states_in = (0..size_states_in-1)
        @range_states_out = (0..size_states_out-1)

        @inputs = inputs ? inputs.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_in)
        @outputs = outputs ? outputs.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_out)
        
        # @hidden_first = hidden_first ? hidden_first.as(HiddenLayerGeneric) : HiddenLayerGeneric.new(@inputs, @size_time_cols, @size_states_in, @size_states_out)
      end

      # def initialize(@size_time_cols = 5, @size_states_in = 3, @size_states_out = 4, inputs = nil, outputs = nil)
      #   @range_time_cols = (0..size_time_cols-1)
      #   @range_states_in = (0..size_states_in-1)
      #   @range_states_out = (0..size_states_out-1)
      #   @inputs = inputs ? inputs.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_in)
      #   @outputs = outputs ? outputs.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_out)
      # end

      def init_simple_nodes(range_time_cols,size_states)
        range_time_cols.map { |tc| size_states.map { |st| 0.0 } }
      end
    end
    
    # structure HiddenChannel
    #   property inputs : SimpleNodes
    #   property outputs : SimpleNodes

    #   property outputs : SimpleNodes

    #   def initialize(@inputs, @outputs)
    #   end
    # end
    
    # structure HiddenLayerGeneric
    #   # has multiple channels
    #   include NetGeneric

    #   property inputs : SimpleNodes

    #   property past_nodes : SimpleNodes
    #   property local_nodes : SimpleNodes
    #   property future_nodes : SimpleNodes

    #   getter range_states_past : Range(Int32,Int32)
    #   getter range_states_local : Range(Int32,Int32)
    #   getter range_states_future : Range(Int32,Int32)

    #   # property combo_nodes : SimpleNodes # aka outputs

    #   def initialize(@inputs, @size_states_out = 4, outputs = nil, past_nodes = nil, local_nodes = nil, future_nodes = nil)
    #     @size_time_cols = @inputs.size
    #     @size_states_in = @inputs.first.size

    #     @range_time_cols = (0..size_time_cols-1)
        
    #     @range_states_in = (0..size_states_in-1)
    #     @range_states_out = (0..size_states_out-1)

    #     @range_states_past = (0..size_states_past-1)
    #     @range_states_local = (0..size_states_local-1)
    #     @range_states_future = (0..size_states_future-1)

    #     # @inputs = inputs ? inputs.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_in)
    #     @outputs = outputs ? outputs.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_out)

    #     @past_nodes = past_nodes ? past_nodes.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_past)
    #     @local_nodes = local_nodes ? local_nodes.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_local)
    #     @future_nodes = future_nodes ? future_nodes.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_future)
    #   end

    #   def initialize(size_time_cols = 5, size_states_in = 3, size_states_out = 4, inputs = nil, outputs = nil) # , hidden_first = nil)
    #     @size_time_cols = inputs ? inputs.as(SimpleNodes).size : size_time_cols
    #     @size_states_in = inputs ? inputs.as(SimpleNodes).first.size : size_states_in
    #     @size_states_out = outputs ? outputs.as(SimpleNodes).first.size : size_states_out

    #     @range_time_cols = (0..size_time_cols-1)
    #     @range_states_in = (0..size_states_in-1)
    #     @range_states_out = (0..size_states_out-1)

    #     @inputs = inputs ? inputs.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_in)
    #     @outputs = outputs ? outputs.as(SimpleNodes) : init_simple_nodes(range_time_cols,range_states_out)
        
    #     # @hidden_first = hidden_first ? hidden_first.as(HiddenLayerGeneric) : HiddenLayerGeneric.new(@inputs, @size_time_cols, @size_states_in, @size_states_out)
    #   end
    # end

    # structure HiddenFirst
    #   property inputs : SimpleNodes
    #   property outputs : SimpleNodes

    #   def initialize(@inputs, @outputs)
    #   end
    # end

    # structure HiddenOther
    #   property inputs : SimpleNodes
    #   property outputs : SimpleNodes
    # end

  end
end

