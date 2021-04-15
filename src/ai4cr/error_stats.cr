module Ai4cr
  class ErrorStats
    include ::JSON::Serializable

    # Must init @score, so set it big enough but not too big (so ErrorStats works w/ to/from JSON)
    # INITIAL_SCORE = Float64::HIGH_ENOUGH_FOR_NETS # Float64::MAX ** (1.0/16)

    HISTORY_SIZE_DEFAULT = 2
    DISTANCE_DEFAULT     = Math.sqrt(Math.sqrt(Float64::MAX)) # .round(14) # 100.0 # Math.sqrt(Float64::HIGH_ENOUGH_FOR_NETS) # DISTANCE_MAX # Float64::MAX # -1.0
    SCORE_DEFAULT        = DISTANCE_DEFAULT                   # 100.0

    getter history_size : Int32
    getter distance : Float64
    getter history : Array(Float64)
    getter score : Float64
    getter hist_correct_plot = Array(String).new # ["tbd"]
    getter hist_output_str_matches = Array(Int32).new

    # DISTANCE_MAX = Float64::MAX / (2**10)

    def initialize(history_size = HISTORY_SIZE_DEFAULT)
      @history_size = case
                      when history_size.nil?
                        HISTORY_SIZE_DEFAULT
                      when history_size < 0
                        raise "Invalid history_size; must be positive."
                      else
                        history_size
                      end

      @distance = DISTANCE_DEFAULT
      @history = Array(Float64).new(history_size)

      # lowest score is best; negatives are effectively invalid
      @score = SCORE_DEFAULT # Float64::HIGH_ENOUGH_FOR_NETS # INITIAL_SCORE
    end

    def distance=(value)
      raise "Invalid value" if value < 0.0
      @distance = Float64.cap_extremes(value, alt_nan: Float64::HIGH_ENOUGH_FOR_NETS, alt_infin_pos: Float64::HIGH_ENOUGH_FOR_NETS, alt_infin_neg: 0.0)
      update_history

      @distance
    end

    def plot_error_distance_history(
      min = 0.0,
      max = 1.0,
      precision = 2.to_i8,
      in_bw = false,
      prefixed = false,
      reversed = false
    )
      # hist = history.map { |h| Float64.cap_extremes(h, alt_nan: 100.0, alt_infin_pos: 100.0, alt_infin_neg: 100.0) }
      hist = history.map { |h| Float64.cap_extremes(h, alt_nan: 100.0, alt_infin_pos: 100.0, alt_infin_neg: 0.0) }
      charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
      charter.plot(hist, prefixed)
    end

    def update_output_str_matches(output_str_matches = [0])
      if @hist_output_str_matches.size < @history_size # - 1
        # Array not 'full' yet, so add latest value to end
        @hist_output_str_matches << output_str_matches
      else
        # Array 'full', so rotate end to front and then put new value at last index
        @hist_output_str_matches.rotate!
        @hist_output_str_matches[-1] = output_str_matches
      end

      # qty_correct = output_str_matches.sum
      # tc_size = output_str_matches.size
      # percent_correct = 100.0 * qty_correct / tc_size
      # # list << qty_correct
      # correct_plot = CHARTER.plot(output_str_matches, false)
      # member.error_stats.update_history_correct_plot(correct_plot)

      @hist_output_str_matches
      # qty_correct
    end

    def update_history_correct_plot(which_correct_plot = "(tbd)")
      if @hist_correct_plot.size < @history_size # - 1
        # Array not 'full' yet, so add latest value to end
        @hist_correct_plot << which_correct_plot
      else
        # Array 'full', so rotate end to front and then put new value at last index
        @hist_correct_plot.rotate!
        @hist_correct_plot[-1] = which_correct_plot
      end

      @hist_correct_plot
    end

    private def update_history
      if @history.size < @history_size # - 1
        # Array not 'full' yet, so add latest value to end
        @history << @distance
      else
        # Array 'full', so rotate end to front and then put new value at last index
        @history.rotate!
        @history[-1] = @distance
      end

      @score = history.map_with_index do |e, i|
        e / (2.0 ** (@history.size - i))
      end.sum

      @history
    end
  end
end
