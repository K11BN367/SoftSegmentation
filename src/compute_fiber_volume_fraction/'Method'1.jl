function compute_fiber_volume_fraction(array)
    Fiber_Sum = sum(array[:, :, 1])
    Epoxy_Sum = sum(array[:, :, 3])
    Empty_Sum = sum(array[:, :, 2])
    Sum = Fiber_Sum + Epoxy_Sum + Empty_Sum
    return (Fiber_Sum / Sum, Epoxy_Sum / Sum, Empty_Sum / Sum)
    #return (Fiber_Sum, Epoxy_Sum, Empty_Sum)
end