import numpy


def _rotation( x, y, z, yaw, pitch, roll):
    input = numpy.array([x, y, z])

    yaw   = _rnd(-r_max, r_max)*numpy.PI/180.0
    pitch = _rnd(-r_max, r_max)*numpy.PI/180.0
    roll  = _rnd(-r_max, r_max)*numpy.PI/180.0

    r = _rotation_matrix(yaw, pitch, roll)

    result = numpy.matmul(r, input)

    return result[0], result[1], result[2]

def _rotation_matrix( yaw, pitch, roll):
    rx = numpy.zeros((3, 3))
    rx[0][0] = 1.0
    rx[1][1] = numpy.cos(yaw)
    rx[1][2] = -numpy.sin(yaw)
    rx[2][1] =  numpy.sin(yaw)
    rx[2][2] =  numpy.cos(yaw)

    ry = numpy.zeros((3, 3))
    ry[0][0] = numpy.cos(pitch)
    ry[0][2] = numpy.sin(pitch)
    ry[1][1] = 1.0
    ry[2][0] = -numpy.sin(pitch)
    ry[2][2] = numpy.cos(pitch)

    rz = numpy.zeros((3, 3))
    rz[0][0] = numpy.cos(roll)
    rz[0][1] = -numpy.sin(roll)
    rz[1][0] = numpy.sin(roll)
    rz[1][1] = numpy.cos(roll)
    rz[2][2] = 1.0

    result = numpy.matmul(numpy.matmul(rz, ry), rx)
    return result


input = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

yaw   = 45.0*numpy.pi/180.0
pitch = 45.0*numpy.pi/180.0
roll  = 45.0*numpy.pi/180.0

r = _rotation_matrix(yaw, pitch, roll)

result = numpy.matmul(r, input)


print(numpy.round(result, 3))