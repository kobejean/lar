#include <iostream>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "lar/core/utils/json.h"
#include "lar/core/map.h"
#include "lar/service/pathfinding_service.h"

using namespace std;

int main(int argc, const char* argv[]) {
    string map_path = "./output/aizu-park-map/map.json";
    string address = "0.0.0.0:50051";

    if (argc > 1) map_path = argv[1];
    if (argc > 2) address = argv[2];

    // Load map from JSON
    cout << "Loading map from: " << map_path << endl;
    ifstream map_ifs(map_path);
    if (!map_ifs.is_open()) {
        cerr << "Error: Could not open map file: " << map_path << endl;
        return 1;
    }

    nlohmann::json map_data = nlohmann::json::parse(map_ifs);
    lar::Map map = map_data;
    cout << "Loaded map with " << map.anchors.size() << " anchors and "
         << map.edges.size() << " edge lists" << endl;

    // Enable gRPC reflection for grpcurl/debugging
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    // Start gRPC server
    lar::PathfindingServiceImpl service(map);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    unique_ptr<grpc::Server> server = builder.BuildAndStart();
    cout << "Server listening on " << address << endl;
    server->Wait();

    return 0;
}
