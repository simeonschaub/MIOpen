def rocmnode(name) {
    return '(rocmtest || miopen) && (' + name + ')'
}

//def miopenCheckout()
//{
//    checkout([
//        $class: 'GitSCM',
//        branches: scm.branches,
//        doGenerateSubmoduleConfigurations: true,
//        extensions: scm.extensions + [[$class: 'SubmoduleOption', parentCredentials: true]],
 //       userRemoteConfigs: scm.userRemoteConfigs
 //   ])
//}

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        lsb_release -sd
        uname -r
        cat /sys/module/amdgpu/version
        ls /opt/ -la
    """
}

//default
// CXX=/opt/rocm/llvm/bin/clang++ CXXFLAGS='-Werror' cmake -DMIOPEN_GPU_SYNC=Off -DCMAKE_PREFIX_PATH=/usr/local -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release ..
//
def cmake_build(Map conf=[:]){

    def compiler = conf.get("compiler","/opt/rocm/llvm/bin/clang++")
    def make_targets = conf.get("make_targets","check")
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined -Wno-option-ignored " + conf.get("extradebugflags", "")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/opt/rocm")
    def build_type_debug = (conf.get("build_type",'release') == 'debug')

    def mlir_args = " -DMIOPEN_USE_MLIR=" + conf.get("mlir_build", "ON")
    // WORKAROUND_ISSUE_3192 Disabling MLIR for debug builds since MLIR generates sanitizer errors.
    if (build_type_debug)
    {
        mlir_args = " -DMIOPEN_USE_MLIR=OFF"
    }

    def setup_args = mlir_args + " -DMIOPEN_GPU_SYNC=Off " + conf.get("setup_flags","")
    def build_fin = conf.get("build_fin", "OFF")

    setup_args = setup_args + " -DCMAKE_PREFIX_PATH=${prefixpath} "

    //cmake_env can overwrite default CXX variables.
    def cmake_envs = "CXX=${compiler} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")

    def package_build = (conf.get("package_build","") == "true")

    if (package_build == true) {
        make_targets = "miopen_gtest package miopen_gtest_check"
        setup_args = " -DMIOPEN_TEST_DISCRETE=OFF " + setup_args
    }

    def miopen_install_path = "${env.WORKSPACE}/install"
    if(conf.get("build_install","") == "true")
    {
        make_targets = 'install ' + make_targets
        setup_args = " -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=${miopen_install_path}" + setup_args
    } else if(package_build == true) {
        setup_args = ' -DBUILD_DEV=Off' + setup_args
    } else {
        setup_args = ' -DBUILD_DEV=On' + setup_args
    }

    // test_flags = ctest -> MIopen flags
    def test_flags = conf.get("test_flags","")

    if (conf.get("vcache_enable","") == "true"){
        def vcache = conf.get(vcache_path,"/var/jenkins/.cache/miopen/vcache")
        build_envs = " MIOPEN_VERIFY_CACHE_PATH='${vcache}' " + build_envs
    } else{
        test_flags = " --disable-verification-cache " + test_flags
    }

    if(build_type_debug){
        setup_args = " -DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'" + setup_args
    }else{
        setup_args = " -DCMAKE_BUILD_TYPE=release" + setup_args
    }

    if(test_flags != ""){
       setup_args = "-DMIOPEN_TEST_FLAGS='${test_flags}'" + setup_args
    }

    if(conf.containsKey("find_mode"))
    {
        def fmode = conf.get("find_mode", "")
        setup_args = " -DMIOPEN_DEFAULT_FIND_MODE=${fmode} " + setup_args
    }
    if(env.CCACHE_HOST)
    {
        setup_args = " -DCMAKE_CXX_COMPILER_LAUNCHER='ccache' -DCMAKE_C_COMPILER_LAUNCHER='ccache' " + setup_args
    }

    if ( build_fin == "ON" )
    {
        setup_args = " -DMIOPEN_INSTALL_CXX_HEADERS=On " + setup_args
    }

    def pre_setup_cmd = """
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            rm -rf install
            mkdir install
            rm -f src/kernels/*.ufdb.txt
            rm -f src/kernels/miopen*.udb
            cd build
        """
    def setup_cmd = conf.get("setup_cmd", "${cmake_envs} cmake ${setup_args}   .. ")
    // WORKAROUND_SWDEV_290754
    // It seems like this W/A is not required since 4.5.
    def build_cmd = conf.get("build_cmd", "LLVM_PATH=/opt/rocm/llvm ${build_envs} dumb-init make -j\$(nproc) ${make_targets}")
    def execute_cmd = conf.get("execute_cmd", "")

    def cmd = conf.get("cmd", """
            ${pre_setup_cmd}
            ${setup_cmd}
            ${build_cmd}
        """)

    if ( build_fin == "ON" )
    {
        def fin_build_cmd = cmake_fin_build_cmd(miopen_install_path)
        cmd += """
            export RETDIR=\$PWD
            cd ${env.WORKSPACE}/fin
            ${fin_build_cmd}
            cd \$RETDIR
        """
    }

    cmd += """
        ${execute_cmd}
    """

    echo cmd
    sh cmd

    // Only archive from master or develop
    if (package_build == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master" ||
        params.PERF_TEST_ARCHIVE == true)) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
        archiveArtifacts artifacts: "build/*.rpm", allowEmptyArchive: true, fingerprint: true
        stash includes: "build/*tar.gz", name: 'miopen_tar'
    }
}

def cmake_fin_build_cmd(prefixpath){
    def flags = "-DCMAKE_INSTALL_PREFIX=${prefixpath} -DCMAKE_BUILD_TYPE=release"
    def compiler = 'clang++'
    def make_targets = "install"
    def compilerpath = "/opt/rocm/llvm/bin/" + compiler
    def configargs = ""
    if (prefixpath != "")
    {
        configargs = "-DCMAKE_PREFIX_PATH=${prefixpath}"
    }

    def fin_cmd = """
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            cd build
            CXX=${compilerpath} cmake ${configargs} ${flags} ..
            dumb-init make -j\$(nproc) ${make_targets}
    """
    return fin_cmd
}

def getDockerImageName(dockerArgs)
{
    checkout scm
    sh "echo ${dockerArgs} > factors.txt"
    def image = "${env.MIOPEN_DOCKER_IMAGE_URL}"
    sh "md5sum Dockerfile requirements.txt dev-requirements.txt >> factors.txt"
    def docker_hash = sh(script: "md5sum factors.txt | awk '{print \$1}' | head -c 6", returnStdout: true)
    sh "rm factors.txt"
    echo "Docker tag hash: ${docker_hash}"
    image = "${image}:ci_${docker_hash}"
    if(params.DOCKER_IMAGE_OVERRIDE && !params.DOCKER_IMAGE_OVERRIDE.empty)
    {
        echo "Overriding the base docker image with ${params.DOCKER_IMAGE_OVERRIDE}"
        image = "${params.DOCKER_IMAGE_OVERRIDE}"
    }
    return image

}

def getDockerImage(Map conf=[:])
{
    checkout scm
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/opt/rocm") // one image for each prefix 1: /usr/local 2:/opt/rocm
    def gpu_arch = "gfx908;gfx90a;gfx942;gfx1100;1201" // prebuilt dockers should have all the architectures enabled so one image can be used for all stages
    def mlir_build = conf.get("mlir_build", "ON") // always ON
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${prefixpath} --build-arg GPU_ARCHS='\"${gpu_arch}\"' --build-arg USE_MLIR='${mlir_build}' "
    //def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCHS='\"${gpu_arch}\"' --build-arg USE_MLIR='${mlir_build}' "
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

def buildHipClangJob(Map conf=[:]){
        show_node_info()
        //miopenCheckout()
        checkout scm
        env.HSA_ENABLE_SDMA=0
        env.DOCKER_BUILDKIT=1
        def image
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1"
        }

        def variant = env.STAGE_NAME

        def needs_gpu = conf.get("needs_gpu", true)
        def lfs_pull = conf.get("lfs_pull", false)

        def retimage
        gitStatusWrapper(credentialsId: "${env.miopen_git_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCm', repo: 'MIOpen') {
            try {
                (retimage, image) = getDockerImage(conf)
                if (needs_gpu) {
                    withDockerContainer(image: image, args: dockerOpts) {
                        timeout(time: 5, unit: 'MINUTES')
                        {
                            sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                        }
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            catch(Exception ex) {
                (retimage, image) = getDockerImage(conf)
                if (needs_gpu) {
                    withDockerContainer(image: image, args: dockerOpts) {
                        timeout(time: 5, unit: 'MINUTES')
                        {
                            sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                        }
                    }
                }
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 420, unit:'MINUTES')
                {
                    if (lfs_pull) {
                        sh "git lfs pull --exclude="
                    }

                    cmake_build(conf)
                }
            }
        }
        return retimage
}

def reboot(){
    build job: 'reboot-slaves', propagate: false , parameters: [string(name: 'server', value: "${env.NODE_NAME}"),]
}

def buildHipClangJobAndReboot(Map conf=[:]){
    try{
        buildHipClangJob(conf)
        cleanWs()
    }
    catch(e){
        echo "throwing error exception for the stage"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (conf.get("needs_reboot", true)) {
            reboot()
        }
    }
}


def CheckPerfDbValid(Map conf=[:]){
    def pdb_image = buildHipClangJob(conf)
    pdb_image.inside(){
        dir(path: "$WORKSPACE"){
            sh "ls install/bin/"
            sh "MIOPEN_LOG_LEVEL=4 LD_LIBRARY_PATH='install/lib:/opt/rocm/lib/' install/bin/fin -i fin/tests/pdb_check_all.json -o pdb_valid_err.json"
            archiveArtifacts "pdb_valid_err.json"
            sh "grep clear pdb_valid_err.json"
            def has_error = sh (
                script: "echo \$?",
                returnStdout: true
            ).trim()
            assert has_error.toInteger() == 0
        }
    }
}
def evaluate(params)
{
  (build_args, partition) = getBuildArgs()
  def tuna_docker_name = getDockerName("${backend}")

  docker.withRegistry('', "$DOCKER_CRED"){
    def tuna_docker
    tuna_docker = docker.build("${tuna_docker_name}", "${build_args} ." )
    tuna_docker.push()
  }

  env_list = params.env.split(' ')
  for(item in env_list)
  {
    if(item.replaceAll("\\s","") != "")
    {
      if(item.contains("="))
      {
        docker_args += " -e ${item}"
      }
      else
      {
        error "Not added to env: ${item}"
      }
    }
  }
  def eval_cmd = ''
  if(params.stage == 'fin_find')
  {
    eval_cmd = '--fin_steps miopen_find_eval'
  }
  else
  {
    eval_cmd = '--fin_steps miopen_perf_eval'
  }
  if(params.dynamic_solvers_only)
  {
    eval_cmd += ' --dynamic_solvers_only'
  }

  sh "docker run ${docker_args} ${tuna_docker_name} python3 /tuna/tuna/go_fish.py miopen ${eval_cmd} --session_id ${params.session_id} --enqueue_only &"
  sh "srun --no-kill -p ${partition} -N 1-10 -l bash -c 'echo ${env.CREDS_PSW} | HOME=/home/slurm docker login -u ${env.CREDS_USR} --password-stdin && HOME=/home/slurm docker run ${docker_args} ${tuna_docker_name} python3 /tuna/tuna/go_fish.py miopen ${eval_cmd} --session_id ${params.session_id}'"
}

def RunPerfTest(Map conf=[:]){
    checkout scm
    def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    try {
        (retimage, image) = getDockerImage(conf)
        withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
        timeout(time: 100, unit: 'MINUTES')
        {
            //unstash 'miopen_tar'
            //sh "tar -zxvf build/miopen-hip-*-Linux-runtime.tar.gz"
            ld_lib="${env.WORKSPACE}/opt/rocm/lib"
            def filename = conf.get("filename", "")
            if(params.PERF_TEST_OVERRIDE != '')
            {
                echo "Appending MIOpenDriver cmd env vars: ${params.PERF_TEST_OVERRIDE}"
                sh "export LD_LIBRARY_PATH=${ld_lib} && ${env.WORKSPACE}/opt/rocm/bin/test_perf.py  --filename ${filename} --install_path ${env.WORKSPACE}/opt/rocm --override ${params.PERF_TEST_OVERRRIDE}"
            }else
            {
                sh "export LD_LIBRARY_PATH=${ld_lib} && ${env.WORKSPACE}/opt/rocm/bin/test_perf.py  --filename ${filename} --install_path ${env.WORKSPACE}/opt/rocm"
            }
            //sh "export LD_LIBRARY_PATH=${ld_lib} && ${env.WORKSPACE}/opt/rocm/bin/test_perf.py  --filename ${filename} --install_path ${env.WORKSPACE}/opt/rocm"
            jenkins_url = "${env.artifact_path}/${env.BRANCH_NAME}/lastSuccessfulBuild/artifact"
            try {
                sh "rm -rf ${env.WORKSPACE}/opt/rocm/bin/old_results/"
                sh "wget -P ${env.WORKSPACE}/opt/rocm/bin/old_results/ ${jenkins_url}/opt/rocm/bin/perf_results/${filename}"
            }
            catch (Exception err){
                currentBuild.result = 'SUCCESS'
            }

            archiveArtifacts artifacts: "opt/rocm/bin/perf_results/${filename}", allowEmptyArchive: true, fingerprint: true
            try{
               sh "${env.WORKSPACE}/opt/rocm/bin/test_perf.py --compare_results --old_results_path ${env.WORKSPACE}/opt/rocm/bin/old_results --filename ${filename}"
            }
            catch (Exception err){
                currentBuild.result = 'SUCCESS'
            }
            cleanWs()
        }
        }
    }
    catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
        echo "The job was cancelled or aborted"
        throw e
    }
}

return this
