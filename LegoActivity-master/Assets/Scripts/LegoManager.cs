using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Photon.Pun;
using Photon.Realtime;

public class LegoManager : MonoBehaviour
{
    private float GridSize = 1.0f;

    // private vars
    private List<GameObject> LegoObjects = new List<GameObject>();
    private int CurrentId = 0;
    private Dictionary<int, Vector3Int> LegoPositions = new Dictionary<int, Vector3Int>();
    private Dictionary<int, Vector3Int> LegoSizes = new Dictionary<int, Vector3Int>();
    private Dictionary<int, int> LegoRotations = new Dictionary<int, int>();
    private Dictionary<int, string> LegoNames = new Dictionary<int, string>();
    private Dictionary<Player, int> BrickOwnership = new Dictionary<Player, int>();
    private List<GameObject> BricksToDelete = new List<GameObject>();

    public GameManager gameManager;

    private bool[,,] Occupied = new bool[100, 100, 100];

    // Start is called before the first frame update
    void Start()
    {
        // Example create brick call
        //CreateBrick("Brick1x4", new Vector3Int(4, 1, 1), new Vector3(1f,0,1f));
        //LoadModel();
    }

    // Update is called once per frame
    void Update()
    {
        // Since we can't delete immediately, try until ownership transfer completes
        TryDestoryBricks();
    }

    // Returns an updated position, and a boolean stating whether the position is valid
    public (Vector3, bool) UpdatePositionBrick(int LegoId, Vector3 newPosition, int newRotation)
    {
        //Vector3 gridPosition = RoundVector3(newPosition);
        // Grid position as integers
        Vector3Int gridPosition = RoundVector3ToInt(newPosition);

        //Debug.Log(gridPosition);

        // All the locations in the grid occupied by this brick (based on rotation, dimensions, position)
        List<Vector3Int> allBrickPositions = GetAllBrickPositions(gridPosition, LegoSizes[LegoId], newRotation);
        List<Vector3Int> allBrickPositionsCurrent = GetAllBrickPositions(LegoPositions[LegoId], LegoSizes[LegoId], LegoRotations[LegoId]);

        // The actual position in the world
        Vector3 worldGridPosition = new Vector3(
            gridPosition.x,
            gridPosition.y * 1.2f,
            gridPosition.z
        );

        // Clear existing spaces occupied by this brick
        foreach (Vector3Int brickPosition in allBrickPositionsCurrent)
        {
            
            Occupied[brickPosition.x, brickPosition.y, brickPosition.z] = false;
        }

        // Check that new position in completely in grid bounds
        foreach (Vector3Int brickPosition in allBrickPositions)
        {
            //Debug.Log(brickPosition);
            if (!Vector3InGrid(brickPosition))
            {
                return (newPosition, false);
            }
        }

        // Check if the new position is occupied by another brick
        foreach (Vector3Int brickPosition in allBrickPositions)
        {
            if (Occupied[brickPosition.x, brickPosition.y, brickPosition.z])
            {
                return (worldGridPosition, false);
            }
        }


        // Check for a possible mid-air placement?
        bool is_midAir = true;
        foreach (Vector3Int brickPosition in allBrickPositions)
        {
            if (brickPosition.y == 0) 
            {
                // On the ground case
                is_midAir = false;
                break;
            }
            else if(Occupied[brickPosition.x, brickPosition.y - 1, brickPosition.z])
            {
                // Atleast one brick position "below" is occupied. So we can place the
                // current brick over it.
                is_midAir = false;
                break;
            }
            else if (Occupied[brickPosition.x, brickPosition.y + 1, brickPosition.z])
            {
                // Atleast one brick position "above" is occupied. So we can place the
                // current brick below it.
                is_midAir = false;
                break;
            }
        }
        // Allow or dont allow based on is mid air? Else allow it to be placed and update.
        if(is_midAir)
        {
            return (worldGridPosition, false); // early return.
        }



        // Place the brick in its new position 
        foreach (Vector3Int brickPosition in allBrickPositions)
        {
            Occupied[brickPosition.x, brickPosition.y, brickPosition.z] = true;
        }

        LegoPositions[LegoId] = gridPosition;
        LegoRotations[LegoId] = newRotation;

        

        return (worldGridPosition, true);
    }

    // Let everyone know if you are moving a brick
    public void TellEveryoneAboutMyBrick(GameObject brick)
    {
        PhotonView photonView = PhotonView.Get(this);
        photonView.RPC("LearnAboutSomeonesBrick", RpcTarget.All, PhotonNetwork.LocalPlayer, PhotonView.Get(brick).ViewID);
    }

    public void TellEveryoneIHaveNoBricks()
    {
        PhotonView photonView = PhotonView.Get(this);
        photonView.RPC("LearnAboutSomeonesBrick", RpcTarget.All, PhotonNetwork.LocalPlayer, -1);
    }

    [PunRPC]
    public void LearnAboutSomeonesBrick(Player player, int brickId)
    {
        // learn where the brick is now
        if(brickId == -1 && BrickOwnership[player] > -1)
        {
            GameObject brick = PhotonView.Find(BrickOwnership[player]).gameObject;
            brick.GetComponent<LegoMove>().UpdatePosition();

            SaveModel();
        }
        BrickOwnership[player] = brickId;
    }

    public bool CanMoveBrick(GameObject brick)
    {
        foreach(KeyValuePair<Player, int> owner in BrickOwnership)
        {
            if (owner.Value == PhotonView.Get(brick).ViewID)
            {
                return false;
            }
        }
        return true;
    }

    public void TellPlayerToGrabBrick(Player player, int brickId)
    {
        PhotonView photonView = PhotonView.Get(this);
        photonView.RPC("PlayerGrabBrick", RpcTarget.All, player, brickId);
    }

    [PunRPC]
    public void PlayerGrabBrick(Player player, int brickId)
    {
        if(player == PhotonNetwork.LocalPlayer)
        {
            Debug.Log(brickId);

            GameObject characterObject = gameManager.LocalPlayerInstance;
            CharacterMovement character = characterObject.GetComponent<CharacterMovement>();
            GameObject brick = PhotonView.Find(brickId).gameObject;
            character.EnterLegoMode(brick, 15.0f);
        }
        
    }

    public void AskMasterToCreateBrick(string modelName)
    {
        PhotonView photonView = PhotonView.Get(this);
        photonView.RPC("MasterCreateBrick", RpcTarget.MasterClient, modelName, PhotonNetwork.LocalPlayer, true);
    }

    [PunRPC]
    public void MasterCreateBrick(string modelName, Player player, bool grab)
    {
        //Debug.Log("Creating Lego!");
        //Vector3 pos = position;
        //pos.y *= 1.2f;
        GameObject brick = CreateBrick(modelName, new Vector3Int(50, 50, 50), 0);
        //brick.name = brick.GetInstanceID().ToString();
        brick.GetComponent<PhotonView>().TransferOwnership(player);


        if (grab)
        {
            // tell everyone this player is moving this brick
            PhotonView photonView = PhotonView.Get(this);
            photonView.RPC("LearnAboutSomeonesBrick", RpcTarget.All, player, PhotonView.Get(brick).ViewID);

        
            TellPlayerToGrabBrick(player, PhotonView.Get(brick).ViewID);
        }
        
    }


    private GameObject CreateBrick(string modelName, Vector3 position, int rotation)
    {
        GameObject legoBrick;
        // Have to make different calls when in a multiplayer room/ in a local testing
        if(PhotonNetwork.IsConnected)
        {
            legoBrick = PhotonNetwork.InstantiateRoomObject(modelName,
                position, Quaternion.AngleAxis(rotation, Vector3.up), 0);
        }
        else
        {
            legoBrick = Instantiate(Resources.Load<GameObject>(modelName), position, Quaternion.identity);
        }
        //legoBrick.GetComponent<LegoMove>().InitializeInLegoMode = true;

        return legoBrick;
    }

    // Remove a Lego brick
    public void DeleteBrick(int LegoId, GameObject brick)
    {
        // De-Register in the Meta Data Dicts or just leave the indexes alone?
        // Cannot null them!
        // LegoSizes[LegoId] = new Vector3Int(0, 0, 0);
        // LegoPositions[LegoId] = new Vector3Int(0, 0, 0);
        // LegoRotations[LegoId] = 0;
        // Delete using the reference
        PhotonView photonView = PhotonView.Get(this);

        List<Vector3Int> allBrickPositionsCurrent = GetAllBrickPositions(LegoPositions[LegoId], LegoSizes[LegoId], LegoRotations[LegoId]);
        foreach (Vector3Int brickPosition in allBrickPositionsCurrent)
        {
            photonView.RPC("TellEveryoneStudGone", RpcTarget.All, brickPosition.x, brickPosition.y, brickPosition.z);
        }

        //GameObject legoBrick = LegoObjects[LegoId];
        //LegoObjects[LegoId] = null;
       
        PhotonView brickView = PhotonView.Get(brick);
        if(!brickView.AmOwner)
        {
            brickView.TransferOwnership(PhotonNetwork.MasterClient);
        }
        photonView.RPC("MasterDestroyBrick", RpcTarget.MasterClient, brickView.ViewID);
        
        // Debug.Log("List size: " + LegoObjects.Count);
    }

    [PunRPC]
    public void MasterDestroyBrick(int brickId)
    {
        GameObject brick = PhotonView.Find(brickId).gameObject;

        // Can't delete immediately, ownership transfer takes time
        BricksToDelete.Add(brick);
    }

    private void TryDestoryBricks()
    {
        for(int i = 0; i < BricksToDelete.Count; i++)
        { 
            GameObject brick = BricksToDelete[i];
            if(brick.GetPhotonView().AmOwner)
            {
                PhotonNetwork.Destroy(brick);
                BricksToDelete.Remove(brick);
                i--;
            }
        }
        
    }

    [PunRPC]
    public void TellEveryoneStudGone(int x, int y, int z)
    {
        Occupied[x, y, z] = false;
    }

    // Register a brick and get its ID
    public int RegisterBrick(Vector3Int dimensions, Vector3Int position, int rotation, string name, GameObject legoBrick)
    {
        int LegoId = CurrentId++;
        LegoSizes[LegoId] = dimensions;
        LegoPositions[LegoId] = position;
        LegoRotations[LegoId] = rotation;
        LegoNames[LegoId] = name;
        //LegoObjects.Add(legoBrick);

        List<Vector3Int> allBrickPositions = GetAllBrickPositions(LegoPositions[LegoId], LegoSizes[LegoId], LegoRotations[LegoId]);

        foreach (Vector3Int brickPosition in allBrickPositions)
        {
            Occupied[brickPosition.x, brickPosition.y, brickPosition.z] = true;
        }

        return LegoId;
    }

    // Find all grid locations the block covers (i.e. and 1x4 brick occupies 4 different grid locations)
    // Coordinates in this function are indices into the Occupied array
    List<Vector3Int> GetAllBrickPositions(Vector3Int originPosition, Vector3Int dimensions, int rotation)
    {
        List<Vector3Int> LocationsList = new List<Vector3Int>();

        // We need to figure out which way this block is rotated
        // Rotate clockwise!
        int xStart, zStart, xEnd, zEnd;
        //Debug.Log(rotation);
        if (rotation == 0)
        {
            xStart = -dimensions.x + 1;
            xEnd = 1;
            zStart = -dimensions.z + 1;
            zEnd = 1;
        }
        else if (rotation == 90)
        {
            xStart = -dimensions.z + 1;
            xEnd = 1;
            zStart = 0;
            zEnd = dimensions.x;
        }
        else if (rotation == 180)
        {
            xStart = 0;
            xEnd = dimensions.x;
            zStart = 0;
            zEnd = dimensions.z;
        }
        else
        {
            xStart = 0;
            xEnd = dimensions.z;
            zStart = -dimensions.x + 1;
            zEnd = 1;
        }

        for (int x = xStart; x < xEnd; x++)
        {
            for (int z = zStart; z < zEnd; z++)
            {
                for (int y = 0; y < dimensions.y; y++)
                {
                    LocationsList.Add(new Vector3Int(
                        originPosition.x + x,
                        originPosition.y + y,
                        originPosition.z + z
                    ));
                }
            }
        }

        return LocationsList;
    }

    // Figures out where the brick is the grid based on its world cooridinates
    Vector3Int RoundVector3ToInt(Vector3 vector)
    {
        return new Vector3Int(
            (int)Round(vector.x, 1.0f),
            (int)Round(vector.y / 1.2f, 1.0f),
            (int)Round(vector.z, 1.0f)
        );
    }

    // Checks whether a vector is within the bounds of the grid
    bool Vector3InGrid(Vector3Int vector)
    {
        return vector.x >= 0
            && vector.x < 100
            && vector.y >= 0
            && vector.y < 100
            && vector.z >= 0
            && vector.z < 100;
    }

    // Rounds a float to a multiple of another number
    float Round(float number, float multiple)
    {
        return Mathf.Ceil(number / multiple) * multiple;
    }

    public void SaveModel()
    {
        // save lego positions, lego name, lego rotations 
        string legoPositionsStr = DictionaryToString<Vector3Int>(LegoPositions);
        string legoRotationsStr = DictionaryToString<int>(LegoRotations);
        string legoNamesStr = DictionaryToString<string>(LegoNames);

        string delimiter = "\r";
        string data = legoPositionsStr + delimiter + legoRotationsStr + delimiter + legoNamesStr;

        PlayerPrefs.SetString("Test", data);
    }

    public void LoadModel()
    {
        char delimiter = '\r';
        string data = PlayerPrefs.GetString("Test");

        Debug.Log(data);

        string[] parts = data.Split(delimiter);

        Dictionary<int, Vector3Int> positions = StringToDictionary<Vector3Int>(parts[0]);
        Dictionary<int, int> rotations = StringToDictionary<int>(parts[1]);
        Dictionary<int, string> names = StringToDictionary<string>(parts[2]);

        for(int i = 0; i < positions.Count; i++ )
        {
            //MasterCreateBrick(names[i], positions[i], rotations[i], PhotonNetwork.LocalPlayer, false);
        }
    }

    public string DictionaryToString<T>(Dictionary<int, T> dict)
    {
        string result = "";
        foreach (KeyValuePair<int, T> pair in dict)
        {
            result += pair.Key + "\t";
            if(pair.Value is int)
            {
                result += pair.Value;
            }
            else if(pair.Value is Vector3Int)
            {
                Vector3Int value = (Vector3Int)(object)pair.Value;
                result += value.x + " " + value.y + " " + value.z;
            }
            else if(pair.Value is string)
            {
                result += pair.Value;
            }
            result += "\n";
        }
        return result.Substring(0, result.Length - 1);
    }

    public Dictionary<int, T> StringToDictionary<T>(string str)
    {
        Dictionary<int, T> result = new Dictionary<int, T>();
        string[] parts = str.Split('\n');
        foreach(string part in parts)
        {
            string[] keyValue = part.Split('\t');
            int key = int.Parse(keyValue[0]);
            if(typeof(T) == typeof(int))
            {
                result.Add(key, (T)(object)int.Parse(keyValue[1]));
            }
            else if(typeof(T) == typeof(Vector3Int))
            {
                string[] components = keyValue[1].Split(' ');
                result.Add(key, (T)(object)new Vector3Int(
                    int.Parse(components[0]),
                    int.Parse(components[1]),
                    int.Parse(components[2])
                ));
            }
            else if(typeof(T) == typeof(string))
            {
                result.Add(key, (T)(object)keyValue[1]);
            }
        }

        return result;
    }

}
