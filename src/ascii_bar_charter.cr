class AsciiBarCharter
  BAR_CHARS = ['_','\u2581','\u2582','\u2583','\u2584','\u2585','\u2586','\u2587','\u2588','\u2589','\u258A']
  BAR_STEP_QTY = BAR_CHARS.size - 1 # first char is a 'floor', not a 'step'
  BAR_COLORS = [:blue,:green,:green,:light_green,:light_green,:yellow,:yellow,:light_red,:light_red,:red,:red]

  getter min, max, min_max_delta : Float64, round_qty

  def initialize(@min : Float64, @max : Float64, @round_qty : Int8)
    @min_max_delta = 1.0 * (max - min)
  end

  def plot(data)
    data.map do |single_data|
      # i = (BAR_STEP_QTY * (d - min) / min_max_delta).round.to_i
      # i = 0 if i < 0
      # i = BAR_STEP_QTY if i > BAR_STEP_QTY
      # bar_chars[i].colorize.fore(bar_colors[i]).back(:light_gray)
      colored(single_data)
    end.join
  end

  def colored(single_data, as_bar = true)
    i = (BAR_STEP_QTY * (single_data - min) / min_max_delta).round.to_i
    i = 0 if i < 0
    i = BAR_STEP_QTY if i > BAR_STEP_QTY
    bar = BAR_CHARS[i].colorize.fore(BAR_COLORS[i]).back(:light_gray)
    # text = as_bar ? BAR_CHARS[i] : single_data.round(round_qty).to_s
    # text.colorize.fore(BAR_COLORS[i]).back(:light_gray)
    # text.colorize.fore(BAR_COLORS[i]).back(:dark_gray)
    as_bar ? bar : (single_data.round(round_qty).to_s + bar.to_s)
  end

  def colored_value(single_data)
    colored(single_data, as_bar = false)
  end
end
