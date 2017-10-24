#!/usr/bin/env python
import rospy
import oct_15_demo.srv
from geometry_msgs.msg import Pose2D, PoseStamped, WrenchStamped
from force_sensor_gateway.msg import ForceSensorData
import yaml
from dvrk import psm
import PyKDL
import numpy as np
from uvtoworld import makeTexturedObjData
from uvtoworld import UVToWorldConverter
from tf_conversions import posemath
from clean_resource_path import cleanResourcePath
import force_sensor_gateway.ransac as ransac
from IPython import embed
from sensor_msgs.msg import RegionOfInterest
from IPython import embed

import os.path
functionPath = os.path.dirname(os.path.realpath(__file__))

def resolvedRates(config,currentPose,desiredPose):
    # compute pose error (result in kdl.twist format)
    poseError = PyKDL.diff(currentPose,desiredPose)
    posErrNorm = poseError.vel.Norm()
    rotErrNorm = poseError.rot.Norm()

    angVelMag = config['angVelMax']
    if rotErrNorm < config['tolRot']:
        angVelMag = 0.0
    elif rotErrNorm < angVelMag:
        angVelMag = rotErrNorm


    # compute velocity magnitude based on position error norm
    if posErrNorm>config['tolPos']:
        tolPosition = config['tolPos']
        lambdaVel = config['velRatio']
        if posErrNorm>(lambdaVel*tolPosition):
            velMag = config['velMax']
        else:
            velMax = config['velMax']
            velMin = config['velMin']
            velMag = velMin + (posErrNorm - tolPosition) * \
                     (velMax - velMin)/(tolPosition*(lambdaVel-1))
    else:
        velMag = 0.0
    # compute angular velocity based on rotation error norm
    if rotErrNorm>config['tolRot']:
        tolRotation = config['tolRot']
        lambdaRot = config['rotRatio']
        if rotErrNorm>(lambdaRot*tolRotation):
            angVelMag = config['angVelMax']
        else:
            angVelMax = config['angVelMax']
            angVelMin = config['angVelMin']
            angVelMag = angVelMin + (rotErrNorm - tolRotation) * \
                        (angVelMax - angVelMin)/(tolRotation*(lambdaRot-1))
    else:
        angVelMag = 0.0
    # The resolved rates is implemented as Nabil Simaan's notes
    # apply both the velocity and angular velocity in the error pose direction
    desiredTwist = PyKDL.Twist()
    poseError.vel.Normalize() # normalize to have the velocity direction
    desiredTwist.vel = poseError.vel*velMag
    poseError.rot.Normalize() # normalize to have the ang vel direction
    desiredTwist.rot = poseError.rot*angVelMag
    return desiredTwist

def point2DtoUV(pointMsg):
    return (0,0)

def uvToOrganWorldConverter(uv, mesh):
    return (0,0,0), (0,0,0)

def organToRobotFrameConverter(vector, organTransform):
    return vector

def arrayToPyKDLRotation(array):
    x = PyKDL.Vector(array[0][0], array[1][0], array[2][0])
    y = PyKDL.Vector(array[0][1], array[1][1], array[2][1])
    z = PyKDL.Vector(array[0][2], array[1][2], array[2][2])
    return PyKDL.Rotation(x,y,z)

def arrayToPyKDLFrame(array):
    rot = arrayToPyKDLRotation(array)
    pos = PyKDL.Vector(array[0][3],array[1][3],array[2][3])
    return PyKDL.Frame(rot,pos)

def rotationFromVector(vectorDesired):
    ''' Find a PyKDL rotation from a vector using taylor series expansion
    '''
    vector = vectorDesired / vectorDesired.Norm()
    # Cross product of vectorDesired with z vector
    v = PyKDL.Vector(-vector.y(), vector.x(), 0)
    s = v.Norm()

    if s == 0:
        retval = PyKDL.Rotation(vector.z(), 0, 0,
                                0, 1, 0,
                                0, 0,vector.z())
        retval.DoRotZ(np.pi/2)
        return retval


    skew = np.matrix([[   0.0, -v.z(),  v.y()],
                      [ v.z(),    0.0, -v.x()],
                      [-v.y(), v.x(),    0.0]])
    c = vector.z()
    R = np.eye(3) + skew + skew*skew*(1-c)/(s*s);

    kdlRotation = arrayToPyKDLRotation(R.tolist())
    z, y  = kdlRotation.GetEulerZYZ()[0:2]
    print(z,y)
    retval = PyKDL.Rotation()
    retval.DoRotZ(z)
    retval.DoRotY(y)
    retval = PyKDL.Rotation.Rot(retval.UnitZ(), np.pi/2) * retval
    return retval

    # K = np.matrix([[        0.0, -vector.z(),  vector.y()],
    #                [ vector.z(),         0.0, -vector.x()],
    #                [-vector.y(),  vector.x(),        0.0]])

    # xCurrent = np.array([currentPose.M.UnitY().x(),
    #                      currentPose.M.UnitY().y(),
    #                      currentPose.M.UnitY().z()])
    # xDesired = R.transpose().tolist()[1]
    # angle = -np.arccos(np.dot(xCurrent,xDesired))

    # A = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K * K

    # return arrayToPyKDLRotation((A*R).tolist())

class Probe2DServer():
    def __init__(self, cameraTransform, objPath):
        s = rospy.Service('probe2D', oct_15_demo.srv.Probe2D, self.probe2D)
        # Set up subscribers
        self.forceSub = rospy.Subscriber('/force_sensor_topic',
                                         ForceSensorData, self.forceCb)
        # self.forceSub = rospy.Subscriber('/dvrk/PSM2/wrench_body_current',
        #                                  WrenchStamped, self.forceCb)
        self.organPoseSub = rospy.Subscriber('/stereo/registration_pose',
                                             PoseStamped, self.poseCb)
        self.roiSub = rospy.Subscriber('/stiffness_roi',
                                        RegionOfInterest, self.roiCb)
        self.cameraTransform = cameraTransform
        self.organTransform = None
        self.robot = psm('PSM2')

        self.toolOffset = .025 # distance from pinching axle to center of orange nub

        rate = 1000.0
        # TODO make these values not hard coded
        self.rate = rospy.Rate(rate) # 1000hz

        self.resolvedRatesConfig = \
        {   'velMin': 2.0/1000,
            'velMax': 30.0/1000,
            'angVelMin': 1.0/180.0*3.14,
            'angVelMax': 60.0/180.0*3.14,
            'tolPos': 0.1/1000.0, # positional tolerance
            'tolRot': 1.0/180.0*3.14, # rotational tolerance
            'velRatio': 1, # the ratio of max velocity error radius to tolarance radius, this value >1
            'rotRatio': 1,
            'dt': 1.0/rate, # this is the time step of the system. 
                            # if rate=1khz, then dt=1.0/1000. However, 
                            # we don't know if the reality will be the same as desired rate
        }

        # TODO make these not hard-coded
        self.maxDepth = 0.0025 # Meters
        self.maxForce = 800 # Not sure of this?
        self.safeZ = .05 # Safe height above organ in meters
        self.normalDistance = 0.005 # Meters

        self.safeSpot = PyKDL.Frame()
        self.safeSpot.p = PyKDL.Vector(0,0.00,-0.05)
        self.safeSpot.M = rotationFromVector(PyKDL.Vector(0,0,-.1))
        
        self.robot.move(self.safeSpot)
        self.resetZRot()
        self.rate.sleep()
        self.force = None

        # Get obj data
        objData = makeTexturedObjData(objPath)
        self.uvToWorldConverter = UVToWorldConverter(objData)

        self.roi = None
        self.stiffness_img_size = 1000 # in pixels

    def poseCb(self, data):
        self.organTransform = posemath.fromMsg(data.pose)

    def resetZRot(self):
        curr = self.robot.get_current_joint_position()
        # if(abs(curr[3]) > np.pi):
        #     print("RESETTING Z")
        self.robot.move_joint(np.array([curr[0],
                                        curr[1],
                                        curr[2],
                                        0,
                                        curr[4],
                                        curr[5]]))

    def forceCb(self,data):
        # force = [data.wrench.force.x,data.wrench.force.y,data.wrench.force.z]
        # f = np.linalg.norm(force)
        # f = data.wrench.force.z
        # self.force = [f,f,f,f]
        self.force = [data.data1, data.data2, data.data3, data.data4]

    def roiCb(self,data):
        self.roi = data

    def probe2D(self, req):
        if type(self.organTransform) == type(None):
            rospy.logwarn("No organ transform found. Returning negative stiffness")
            return oct_15_demo.srv.Probe2DResponse(-1)
        if type(self.roi) == type(None):
            rospy.logwarn("No ROI. Returning Zero")
            return oct_15_demo.srv.Probe2DResponse(-1)
        # pos = self.organTransform * position
        point2D = np.array([req.Target.x, req.Target.y])
        # Scale and offset based on ROI
        point2D[0] = point2D[0] * self.roi.width + self.roi.x_offset
        point2D[1] = point2D[1] * self.roi.height + self.roi.y_offset
        point2D = point2D / float(self.stiffness_img_size)
        point2D[1] = 1 - point2D[1]
        pos, norm = self.uvToWorldConverter.toWorldSpace(point2D)
        pos = PyKDL.Vector(pos[0],pos[1],pos[2])
        norm = PyKDL.Vector(norm[0],norm[1],norm[2])
        pos = server.cameraTransform * server.organTransform * pos
        norm = server.cameraTransform.M * server.organTransform.M * norm
        if norm.Norm() == 0:
            rospy.logwarn("No valid point on mesh found. Returning negative stiffness")
            return oct_15_demo.srv.Probe2DResponse(-1)
        disp, force = self.probe(pos, norm)
        if type(force) == type(None):
            rospy.logwarn("No forces received. Returning negative stiffness")
            return oct_15_demo.srv.Probe2DResponse(-1)
        stiffness = ransac.fitForceData(force, disp)
        # try:
        #     stiffness = fitForceData(force, disp)
        # except:
        #     rospy.logwarn("Unable to fit stiffness to force measurements. Returning negative stiffness")
        #     return oct_15_demo.srv.Probe2DResponse(-1)
        return oct_15_demo.srv.Probe2DResponse(stiffness)

    def probe(self, position, normal):
        self.resetZRot()
        traj = self.makeTrajectory(position,normal)
        for idx, pose in enumerate(traj):
            print idx
            #TODO: Take care of data, it is not a 6xn array, but
            #instead a 1xn displacement array and 4xn force array.
            if idx == 2:
                displacements, forceData = self.move(pose,self.maxForce)
            else:
                self.move(pose, self.maxForce)

        np.save(os.path.join(functionPath,"force.data"),forceData)
        np.save(os.path.join(functionPath,"displacement.data"),displacements)
        return displacements, forceData

    def makeTrajectory(self, position, normal):
        # Makes a trajectory that constitutes a full probing motion
        normal.Normalize()
        startPosition = position + normal*self.normalDistance
        poses = []
        # Go to a safe position
        pose0 = PyKDL.Frame()
        targetZ = self.safeZ + (self.cameraTransform * self.organTransform).p.z()
        pose0.p = PyKDL.Vector(startPosition.x(), startPosition.y(), targetZ)
        pose0.M = rotationFromVector(normal * -1)
        poses.append(pose0)
        # Lower to desired Z for start of probe
        pose1 = PyKDL.Frame()
        pose1.p = startPosition
        pose1.M = pose0.M
        poses.append(pose1)
        # This is the actual probing
        pose2 = PyKDL.Frame()
        pose2.p = position - normal * self.maxDepth
        pose2.M = pose0.M
        poses.append(pose2)
        # Go back to where we started
        poses.append(pose1)
        poses.append(pose0)

        return poses

    def move(self,desiredPose, maxForce):
        currentPose = self.robot.get_desired_position()
        currentPose.p = currentPose.p
        forceArray = np.empty((0,4), float)
        displacements = np.array([0], float)
        # Remove z rotation
        angle = np.arccos(PyKDL.dot(desiredPose.M.UnitX(), currentPose.M.UnitX()))
        rot = PyKDL.Rotation.Rot(desiredPose.M.UnitZ(), angle)

        # Added offset representing tooltip
        desiredPosition = desiredPose.p - desiredPose.M.UnitZ()*self.toolOffset
        desiredPoseWithOffset = PyKDL.Frame(desiredPose.M, desiredPosition)
        measuredPose_previous = self.robot.get_current_position()
        startForce = self.force[1]
        while not rospy.is_shutdown():
            # get current and desired robot pose (desired is the top of queue)
            # compute the desired twist "x_dot" from motion command
            xDotMotion = resolvedRates(self.resolvedRatesConfig,
                                       currentPose,
                                       desiredPoseWithOffset) # xDotMotion is type [PyKDL.Twist]
            currentPose = PyKDL.addDelta(currentPose,xDotMotion,self.resolvedRatesConfig['dt'])
            
            #Save displacement and force in different arrays
            if type(self.force) == type(None):
                rospy.logwarn("Probe Service: No forces detected. Not moving and returning None")
                return None, None
            data = np.array([self.force])
            forceArray = np.append(forceArray, data, axis = 0)

            if xDotMotion.vel.Norm() <= 0.001 and xDotMotion.rot.Norm() <= 0.1:
                break

            if self.force[1] - startForce > maxForce:
                rospy.logwarn("Max Force Exceeded: " +  str(self.force))
                break
            
            
            self.robot.move(currentPose, interpolate = False)
            self.rate.sleep()
            measuredPose_current = self.robot.get_current_position()
            currentDisplacement = measuredPose_current.p-measuredPose_previous.p
            # currentDisplacement =  xDotMotion.vel.Norm() * self.resolvedRatesConfig['dt']
            currentDisplacement = currentDisplacement.Norm()

            # currentDisplacement = displacements[len(displacements)-1] + currentDisplacement
            displacements = np.append(displacements, [currentDisplacement])
        return displacements.tolist(), forceArray.tolist()

if __name__=="__main__":
    yamlFile = cleanResourcePath("package://dvrk_vision/defaults/registration_params.yaml")
    with open(yamlFile, 'r') as stream:
        data = yaml.load(stream)
    cameraTransform = arrayToPyKDLFrame(data['transform'])
    np.set_printoptions(precision=2)
    # print np.matrix(data['transform'])
    # print cameraTransform.M
    rospy.init_node('probe_2D_server')
    meshPath = rospy.get_param("~mesh_path")
    objPath = cleanResourcePath(meshPath)
    server = Probe2DServer(cameraTransform, objPath)
    # TODO turn off embed
    embed()
    # rospy.spin()