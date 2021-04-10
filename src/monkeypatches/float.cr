struct Float64
  # Float64.avoid_extremes(value)
  def self.avoid_extremes(value : Float64, alt_nan = Float64.new(0), alt_infin_pos = Float64::MAX, alt_infin_neg = -Float64::MAX)
    # For sake of ai4cr-internal calc's and for sake of to/from_json, we need to avoid NaN and Infinity.
    # In some cases, we want to specify alternate values
    case
    when value.nan?
      alt_nan
    when value.infinite? && value > 0
      alt_infin_pos
    when value.infinite? && value < 0
      alt_infin_neg
    else
      value
    end
  end
end
