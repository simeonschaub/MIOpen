def rocmnode(name) {
    checkout scm
    return '(rocmtest || miopen) && (' + name + ')'
}

def getDockerImageName(dockerArgs)
{
    sh "echo ${dockerArgs} > factors.txt"
    def image = "${env.MIOPEN_DOCKER_IMAGE_URL}"
    sh "md5sum Dockerfile requirements.txt dev-requirements.txt >> factors.txt"
    def docker_hash = sh(script: "md5sum factors.txt | awk '{print \$1}' | head -c 6", returnStdout: true)
    sh "rm factors.txt"
    echo "Docker tag hash: ${docker_hash}"
    image = "${image}:ci_${docker_hash}"
    if(params.DOCKER_IMAGE_OVERRIDE != '')
    {
        echo "Overriding the base docker image with ${params.DOCKER_IMAGE_OVERRIDE}"
        image = "${params.DOCKER_IMAGE_OVERRIDE}"
    }
    return image

}

def getDockerImage(Map conf=[:])
{
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/opt/rocm") // one image for each prefix 1: /usr/local 2:/opt/rocm
    def gpu_arch = "gfx908;gfx90a;gfx942;gfx1100" // prebuilt dockers should have all the architectures enabled so one image can be used for all stages
    def mlir_build = conf.get("mlir_build", "ON") // always ON
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${prefixpath} --build-arg GPU_ARCHS='\"${gpu_arch}\"' --build-arg USE_MLIR='${mlir_build}' "
    if(env.CCACHE_HOST)
    {
        def check_host = sh(script:"""(printf "PING\r\n";) | nc -N ${env.CCACHE_HOST} 6379 """, returnStdout: true).trim()
        if(check_host == "+PONG")
        {
            echo "FOUND CCACHE SERVER: ${CCACHE_HOST}"
        }
        else
        {
            echo "CCACHE SERVER: ${CCACHE_HOST} NOT FOUND, got ${check_host} response"
        }
        dockerArgs = dockerArgs + " --build-arg CCACHE_SECONDARY_STORAGE='redis://${env.CCACHE_HOST}' --build-arg COMPILER_LAUNCHER='ccache' "
        env.CCACHE_DIR = """/tmp/ccache_store"""
        env.CCACHE_SECONDARY_STORAGE="""redis://${env.CCACHE_HOST}"""
    }
    echo "Docker Args: ${dockerArgs}"

    def image = getDockerImageName(dockerArgs)

    def dockerImage
    try{
        echo "Pulling down image: ${image}"
        dockerImage = docker.image("${image}")
        dockerImage.pull()
    }
    catch(org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
        echo "The job was cancelled or aborted"
        throw e
    }
    catch(Exception ex)
    {
        dockerImage = docker.build("${image}", "${dockerArgs} .")
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            dockerImage.push()
        }
    }
    return [dockerImage, image]
}

def RunPerfTest(Map conf=[:]){
    checkout scm
    def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    try {
        (retimage, image) = getDockerImage(conf)
        withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
        timeout(time: 100, unit: 'MINUTES')
        {
            sh "echo TEST"
            cleanWs()
        }
        }
    }
    catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
        echo "The job was cancelled or aborted"
        throw e
    }
}
