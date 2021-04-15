struct Float64
  # Looks like the smallest and biggest floats that work w/ to/from json are:
  # * smallest: 1e-307 (1e-308 -> 0.0)
  # * biggest : 1e+308
  # So, to be safe for RELU sake, we need to cap our floats w/in those ranges.

  # These nets don't need huge numbers, so for sake a nets, we can set an artificial max (aka 'HIGH_ENOUGH_FOR_NETS'):
  HIGH_ENOUGH_FOR_NETS = Math.sqrt(Float64::MAX) # 1.0e120 # 1.0e12 # Float64::MAX / (2**10) # MAX - 1

  # Float64.avoid_extremes(value)
  def self.avoid_extremes(value : Float64, alt_nan = Float64.new(0), alt_infin_pos = HIGH_ENOUGH_FOR_NETS, alt_infin_neg = -HIGH_ENOUGH_FOR_NETS)
    # For sake of ai4cr-internal calc's and for sake of to/from_json, we need to avoid NaN and Infinity.
    # In some cases, we want to specify alternate values
    case
    when value.nan?
      alt_nan
    when value > alt_infin_pos # value.infinite? && value > 0
      alt_infin_pos
    when value < alt_infin_neg # value.infinite? && value < 0
      alt_infin_neg
    else
      value
    end
  end

  def self.cap_extremes(value : Float64, alt_nan = Float64.new(0), alt_infin_pos = HIGH_ENOUGH_FOR_NETS, alt_infin_neg = -HIGH_ENOUGH_FOR_NETS)
    # For now, just re-use 'avoid_extremes' internally. Keep to distinguish or remove?
    avoid_extremes(value: value, alt_nan: alt_nan, alt_infin_pos: alt_infin_pos, alt_infin_neg: alt_infin_neg)
  end
end
